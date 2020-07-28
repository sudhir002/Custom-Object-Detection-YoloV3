import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard,ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from Helper.TrainUtils import get_classes, get_anchors, create_model, create_tiny_model, data_generator, data_generator_wrapper, ChangeToOtherMachine

YOLO_filename = "Dataset/Dataset.txt"
YOLO_classname = "Dataset/data_classes.txt"
anchors_path = "Models/yolo_anchors.txt"
weights_path = "Models/yolo.h5"
log_dir = "log_dir/"
val_split = 0.1
is_tiny = False
random_seed = None
epochs = 200
input_shape = (416, 416)

def Training():
    np.random.seed(random_seed)
    class_names = get_classes(YOLO_classname)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

     # multiple of 32, height, width
    epoch1, epoch2 = epochs, epochs
    is_tiny_version = len(anchors) == 6  # default setting
    if is_tiny:
        model = create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2, weights_path=weights_path)
    else:
        model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path=weights_path)  # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(
        "OutputModels/checkpoint.h5",
        monitor="val_loss",
        save_weights_only=True,
        save_best_only=True,
        period=5,
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=20, verbose=1)
    with open(YOLO_filename) as f:
        lines = f.readlines()
    # This step makes sure that the path names correspond to the local machine
    # This is important if annotation and training are done on different machines (e.g. training on AWS)
    lines = ChangeToOtherMachine(lines, remote_machine="")
    np.random.shuffle(lines)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a decent model.
    if True:
        model.compile(
            optimizer=Adam(lr=1e-3),
            loss={"yolo_loss": lambda y_true, y_pred: y_pred},
        )
        batch_size = 16
        print( "Train on {} samples, val on {} samples, with batch size {}.".format(num_train, num_val, batch_size))
        history = model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val // batch_size),
            epochs=epoch1,
            initial_epoch=0,
            callbacks=[logging, checkpoint],
        )
        model.save_weights("OutputModels/wristModel_stage_1.h5")

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is unsatisfactory.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(
            optimizer=Adam(lr=1e-4), loss={"yolo_loss": lambda y_true, y_pred: y_pred}
        )  # recompile to apply the change
        print("Unfreeze all layers.")
        batch_size = (4)
        print("Train on {} samples, val on {} samples, with batch size {}.".format(num_train, num_val, batch_size))
        history = model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val // batch_size),
            epochs=epoch1 + epoch2,
            initial_epoch=epoch1,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping],
        )
        model.save_weights("OutputModels/wristModel_final.h5")

if __name__ == "__main__":
    Training()
