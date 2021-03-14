import tensorflow as tf


class CassavaModel:
    def __init__(self, path_to_saved_model='./inceptionv3_finetuned_whole_model'):
        self.model_path = path_to_saved_model
        self.model = tf.keras.models.load_model(path_to_saved_model)
        self.label_dictionary = {"0": "Cassava Bacterial Blight (CBB)", "1": "Cassava Brown Streak Disease (CBSD)",
                                 "2": "Cassava Green Mottle (CGM)", "3": "Cassava Mosaic Disease (CMD)",
                                 "4": "Healthy"}

    def predict(self, path_to_image):
        with open(path_to_image, 'rb') as fp:
            img_byte = fp.read()
        tensor_img = tf.io.decode_image(img_byte, dtype=tf.float32)
        tensor_img = tf.image.resize(tensor_img, size=(512, 512))

        if tensor_img.numpy().max() < 1.1:
            tensor_img = tf.reshape(tensor_img, (1, 512, 512, 3))
        else:
            tensor_img = tf.reshape(tensor_img, (1, 512, 512, 3)) / 255.0
        prediction = self.model.predict(tensor_img)
        label = self.label_dictionary[str(prediction.argmax())]
        pval = round(prediction.max(), 3)
        return label, pval
