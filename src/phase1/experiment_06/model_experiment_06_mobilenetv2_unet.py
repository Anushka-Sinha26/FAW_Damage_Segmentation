"""
Experiment 06
MobileNetV2 ImageNet Encoder + U-Net Decoder

Purpose:
    Final planned RGB segmentation experiment for the
    Fall Armyworm maize-leaf damage segmentation study.

Input:
    RGB image, 320 x 320 x 3
    Values supplied by dataset: [0, 1]

Output:
    Damage probability mask, 320 x 320 x 1
    Values: [0, 1]

Encoder:
    MobileNetV2 pretrained on ImageNet

Decoder:
    U-Net style multi-scale decoder

Training:
    Stage 1: encoder frozen
    Stage 2: partial encoder fine-tuning

Important:
    This file does not modify the dataset or previous experiments.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 320


# ============================================================
# DICE COEFFICIENT
# ============================================================

def dice_coefficient(
    y_true,
    y_pred,
    smooth=1e-6
):
    """
    Soft Dice coefficient.
    """

    y_true = tf.cast(
        y_true,
        tf.float32
    )

    y_pred = tf.cast(
        y_pred,
        tf.float32
    )

    y_true_flat = tf.reshape(
        y_true,
        [-1]
    )

    y_pred_flat = tf.reshape(
        y_pred,
        [-1]
    )

    intersection = tf.reduce_sum(
        y_true_flat * y_pred_flat
    )

    denominator = (
        tf.reduce_sum(y_true_flat)
        +
        tf.reduce_sum(y_pred_flat)
    )

    dice = (
        2.0 * intersection
        + smooth
    ) / (
        denominator
        + smooth
    )

    return dice


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    y_true,
    y_pred
):
    """
    Dice loss = 1 - Dice.
    """

    return 1.0 - dice_coefficient(
        y_true,
        y_pred
    )


# ============================================================
# BCE + DICE LOSS
# ============================================================

def bce_dice_loss(
    y_true,
    y_pred
):
    """
    Combined loss used for Experiment 06.

    50% Binary Cross Entropy
    50% Dice Loss
    """

    bce = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred
    )

    bce = tf.reduce_mean(
        bce
    )

    dice = dice_loss(
        y_true,
        y_pred
    )

    total_loss = (
        0.5 * bce
        +
        0.5 * dice
    )

    return total_loss


# ============================================================
# DICE METRIC
# ============================================================

def dice_metric(
    y_true,
    y_pred
):
    """
    Validation/training Dice metric.
    """

    return dice_coefficient(
        y_true,
        y_pred
    )


# ============================================================
# IoU METRIC
# ============================================================

def iou_metric(
    y_true,
    y_pred,
    smooth=1e-6
):
    """
    Binary IoU metric using threshold 0.5.
    """

    y_true = tf.cast(
        y_true > 0.5,
        tf.float32
    )

    y_pred = tf.cast(
        y_pred > 0.5,
        tf.float32
    )

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    union = (
        tf.reduce_sum(y_true)
        +
        tf.reduce_sum(y_pred)
        -
        intersection
    )

    iou = (
        intersection
        + smooth
    ) / (
        union
        + smooth
    )

    return iou


# ============================================================
# DECODER BLOCK
# ============================================================

def decoder_block(
    x,
    skip,
    filters,
    name
):
    """
    U-Net decoder block.

    Steps:
        1. Bilinear upsampling
        2. Skip connection
        3. Conv
        4. BatchNorm
        5. ReLU
        6. Conv
        7. BatchNorm
        8. ReLU
    """

    # --------------------------------------------------------
    # Upsampling
    # --------------------------------------------------------

    x = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        name=f"{name}_upsample"
    )(x)

    # --------------------------------------------------------
    # Skip connection
    # --------------------------------------------------------

    if skip is not None:

        x = layers.Concatenate(
            name=f"{name}_concat"
        )(
            [
                x,
                skip
            ]
        )

    # --------------------------------------------------------
    # First convolution
    # --------------------------------------------------------

    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name=f"{name}_conv1"
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_bn1"
    )(x)

    x = layers.ReLU(
        name=f"{name}_relu1"
    )(x)

    # --------------------------------------------------------
    # Second convolution
    # --------------------------------------------------------

    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name=f"{name}_conv2"
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_bn2"
    )(x)

    x = layers.ReLU(
        name=f"{name}_relu2"
    )(x)

    return x


# ============================================================
# BUILD MOBILENETV2 U-NET
# ============================================================

def build_mobilenetv2_unet(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    encoder_trainable=False
):
    """
    Builds MobileNetV2 + U-Net.

    encoder_trainable=False:
        Used for Stage 1.

    encoder_trainable=True:
        Used before applying partial fine-tuning.
    """

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    inputs = layers.Input(
        shape=input_shape,
        name="rgb_input"
    )

    # --------------------------------------------------------
    # Dataset currently produces [0,1].
    #
    # MobileNetV2 preprocessing:
    # [0,1] -> [-1,1]
    #
    # x * 2 - 1
    # --------------------------------------------------------

    x = layers.Rescaling(
        scale=2.0,
        offset=-1.0,
        name="mobilenetv2_preprocessing"
    )(inputs)

    # --------------------------------------------------------
    # MobileNetV2 encoder
    # --------------------------------------------------------

    encoder = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    encoder.trainable = encoder_trainable

    # --------------------------------------------------------
    # Feature extraction layers
    #
    # 320 -> 160 -> 80 -> 40 -> 20 -> 10
    # --------------------------------------------------------

    skip_1 = encoder.get_layer(
        "block_1_expand_relu"
    ).output

    skip_2 = encoder.get_layer(
        "block_3_expand_relu"
    ).output

    skip_3 = encoder.get_layer(
        "block_6_expand_relu"
    ).output

    skip_4 = encoder.get_layer(
        "block_13_expand_relu"
    ).output

    bottleneck = encoder.get_layer(
        "out_relu"
    ).output

    # --------------------------------------------------------
    # Create encoder feature model
    # --------------------------------------------------------

    encoder_model = Model(
        inputs=encoder.input,
        outputs=[
            skip_1,
            skip_2,
            skip_3,
            skip_4,
            bottleneck
        ],
        name="mobilenetv2_encoder"
    )

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    (
        skip_1,
        skip_2,
        skip_3,
        skip_4,
        bottleneck
    ) = encoder_model(x)

    # --------------------------------------------------------
    # Decoder
    #
    # 10 -> 20
    # --------------------------------------------------------

    x = decoder_block(
        bottleneck,
        skip_4,
        filters=256,
        name="decoder_1"
    )

    # --------------------------------------------------------
    # 20 -> 40
    # --------------------------------------------------------

    x = decoder_block(
        x,
        skip_3,
        filters=128,
        name="decoder_2"
    )

    # --------------------------------------------------------
    # 40 -> 80
    # --------------------------------------------------------

    x = decoder_block(
        x,
        skip_2,
        filters=96,
        name="decoder_3"
    )

    # --------------------------------------------------------
    # 80 -> 160
    # --------------------------------------------------------

    x = decoder_block(
        x,
        skip_1,
        filters=64,
        name="decoder_4"
    )

    # --------------------------------------------------------
    # 160 -> 320
    # --------------------------------------------------------

    x = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        name="final_upsample"
    )(x)

    # --------------------------------------------------------
    # Final refinement
    # --------------------------------------------------------

    x = layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="final_refinement"
    )(x)

    # --------------------------------------------------------
    # Damage probability
    # --------------------------------------------------------

    outputs = layers.Conv2D(
        filters=1,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        name="damage_probability"
    )(x)

    # --------------------------------------------------------
    # Complete model
    # --------------------------------------------------------

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="MobileNetV2_UNet_FAWSegmentation"
    )

    return model


# ============================================================
# COMPILE MODEL
# ============================================================

def compile_model(
    model,
    learning_rate
):
    """
    Compile the model with Adam + BCE/Dice loss.
    """

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss=bce_dice_loss,
        metrics=[
            dice_metric,
            iou_metric,
            tf.keras.metrics.BinaryAccuracy(
                name="pixel_accuracy"
            )
        ]
    )

    return model


# ============================================================
# PARTIAL FINE-TUNING
# ============================================================

def set_encoder_fine_tuning(
    model,
    trainable_from_layer=100
):
    """
    Unfreezes only the later MobileNetV2 layers.

    Earlier encoder layers remain frozen.

    This avoids aggressively fine-tuning the entire
    ImageNet encoder on the small FAW dataset.
    """

    encoder = model.get_layer(
        "mobilenetv2_encoder"
    )

    encoder.trainable = True

    for index, layer in enumerate(
        encoder.layers
    ):

        if index < trainable_from_layer:

            layer.trainable = False

        else:

            layer.trainable = True

    return model


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_information(
    model
):
    """
    Returns parameter counts.
    """

    total_params = model.count_params()

    trainable_params = sum(
        int(
            tf.keras.backend.count_params(
                weight
            )
        )
        for weight in model.trainable_weights
    )

    non_trainable_params = sum(
        int(
            tf.keras.backend.count_params(
                weight
            )
        )
        for weight in model.non_trainable_weights
    )

    return {
        "total_parameters": int(
            total_params
        ),
        "trainable_parameters": int(
            trainable_params
        ),
        "non_trainable_parameters": int(
            non_trainable_params
        )
    }


# ============================================================
# CUSTOM OBJECTS
# ============================================================

def get_custom_objects():
    """
    Custom objects required when loading a saved model.
    """

    return {
        "dice_coefficient":
            dice_coefficient,

        "dice_loss":
            dice_loss,

        "bce_dice_loss":
            bce_dice_loss,

        "dice_metric":
            dice_metric,

        "iou_metric":
            iou_metric
    }


# ============================================================
# MODEL BUILD TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("EXPERIMENT 06 - MODEL BUILD TEST")
    print("MobileNetV2 ImageNet Encoder + U-Net Decoder")
    print("=" * 80)

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    model = build_mobilenetv2_unet(
        input_shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        ),
        encoder_trainable=False
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    compile_model(
        model,
        learning_rate=1e-4
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    model.summary()

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    information = get_model_information(
        model
    )

    print("\n")
    print("=" * 80)
    print("MODEL INFORMATION")
    print("=" * 80)

    print(
        f"Total parameters     : "
        f"{information['total_parameters']:,}"
    )

    print(
        f"Trainable parameters : "
        f"{information['trainable_parameters']:,}"
    )

    print(
        f"Non-trainable params : "
        f"{information['non_trainable_parameters']:,}"
    )

    # --------------------------------------------------------
    # Dummy input test
    # --------------------------------------------------------

    dummy_input = tf.random.uniform(
        shape=(
            1,
            IMG_SIZE,
            IMG_SIZE,
            3
        ),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32
    )

    dummy_output = model(
        dummy_input,
        training=False
    )

    print("\n")
    print("=" * 80)
    print("OUTPUT TEST")
    print("=" * 80)

    print(
        "Input shape :",
        dummy_input.shape
    )

    print(
        "Output shape:",
        dummy_output.shape
    )

    print(
        "Output min  :",
        float(
            tf.reduce_min(
                dummy_output
            )
        )
    )

    print(
        "Output max  :",
        float(
            tf.reduce_max(
                dummy_output
            )
        )
    )

    # --------------------------------------------------------
    # Shape assertion
    # --------------------------------------------------------

    expected_shape = (
        1,
        IMG_SIZE,
        IMG_SIZE,
        1
    )

    actual_shape = tuple(
        dummy_output.shape
    )

    if actual_shape != expected_shape:

        raise RuntimeError(
            "MODEL OUTPUT SHAPE ERROR: "
            f"Expected {expected_shape}, "
            f"got {actual_shape}"
        )

    # --------------------------------------------------------
    # Range assertion
    # --------------------------------------------------------

    output_min = float(
        tf.reduce_min(
            dummy_output
        )
    )

    output_max = float(
        tf.reduce_max(
            dummy_output
        )
    )

    if output_min < 0.0 or output_max > 1.0:

        raise RuntimeError(
            "MODEL OUTPUT RANGE ERROR."
        )

    print("\n")
    print("=" * 80)
    print("MODEL BUILD TEST PASSED")
    print("=" * 80)