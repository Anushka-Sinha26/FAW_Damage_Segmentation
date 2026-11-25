"""RGB-thermal fusion U-Net (Phase 3).

RGB branch: identical to Phase 1 Exp 06 / e00 (ImageNet MobileNetV2 encoder, frozen in Stage 1,
skips at 1/2, 1/4, 1/8, 1/16 + 1/32 bottleneck, same decoder and head).
Thermal branch: small CNN trained from scratch on a 2-channel input
  [per-image z-scored apparent temperature (0 outside footprint), footprint-valid mask],
producing features at the same five scales (16, 32, 48, 64, 96 channels).
Fusion at every scale:
  mode "film":   spatial FiLM, F' = F * (1 + gamma(T)) + beta(T); gamma/beta 1x1 convs initialised
                 to zero, so training starts exactly from the RGB-only network.
  mode "concat": F' = concat(F, T).
"""
import tensorflow as tf
from tensorflow.keras import Model, layers

SKIPS = ["block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu", "out_relu"]
TH_CH = [16, 32, 48, 64, 96]


def thermal_encoder(t):
    feats, x = [], t
    for i, f in enumerate(TH_CH):
        for j, s in enumerate((2, 1)):
            x = layers.Conv2D(f, 3, strides=s, padding="same", use_bias=False, name=f"th{i}_conv{j}")(x)
            x = layers.BatchNormalization(name=f"th{i}_bn{j}")(x)
            x = layers.ReLU(name=f"th{i}_relu{j}")(x)
        feats.append(x)
    return feats


def fuse(rgb, th, mode, name):
    if mode == "concat":
        return layers.Concatenate(name=f"{name}_concat")([rgb, th])
    c = rgb.shape[-1]
    g = layers.Conv2D(c, 1, kernel_initializer="zeros", bias_initializer="zeros", name=f"{name}_gamma")(th)
    b = layers.Conv2D(c, 1, kernel_initializer="zeros", bias_initializer="zeros", name=f"{name}_beta")(th)
    g1 = layers.Rescaling(1.0, offset=1.0, name=f"{name}_one_plus_gamma")(g)
    return layers.Add(name=f"{name}_add")([layers.Multiply(name=f"{name}_mul")([rgb, g1]), b])


def build_fusion_unet(M, input_shape=(320, 320, 3), mode="film"):
    rgb_in = layers.Input(shape=input_shape, name="rgb_input")
    th_in = layers.Input(shape=(input_shape[0], input_shape[1], 2), name="thermal_input")
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenetv2_preprocessing")(rgb_in)
    enc = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    enc.trainable = False
    enc_model = Model(enc.input, [enc.get_layer(n).output for n in SKIPS], name="mobilenetv2_encoder")
    s1, s2, s3, s4, bott = enc_model(x)
    t1, t2, t3, t4, t5 = thermal_encoder(th_in)
    s1, s2, s3, s4, bott = (fuse(s1, t1, mode, "fuse1"), fuse(s2, t2, mode, "fuse2"), fuse(s3, t3, mode, "fuse3"),
                            fuse(s4, t4, mode, "fuse4"), fuse(bott, t5, mode, "fuse5"))
    x = M.decoder_block(bott, s4, filters=256, name="decoder_1")
    x = M.decoder_block(x, s3, filters=128, name="decoder_2")
    x = M.decoder_block(x, s2, filters=96, name="decoder_3")
    x = M.decoder_block(x, s1, filters=64, name="decoder_4")
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="final_upsample")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="final_refinement")(x)
    out = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="damage_probability")(x)
    return Model([rgb_in, th_in], out, name=f"MobileNetV2_UNet_RGBT_{mode}")


def build_model(cfg, M, encoder_trainable=False):
    """Build the model named in cfg['model'] (default: RGB-only Exp 06 U-Net)."""
    P = cfg["train"]["patch"]
    mc = cfg.get("model") or {}
    if mc.get("fusion"):
        return build_fusion_unet(M, (P, P, 3), mc["fusion"])
    return M.build_mobilenetv2_unet(input_shape=(P, P, 3), encoder_trainable=encoder_trainable)
