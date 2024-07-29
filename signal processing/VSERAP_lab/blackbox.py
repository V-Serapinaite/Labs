from keras.applications.resnet import ResNet50

from keras.applications.resnet import preprocess_input
import numpy as np
from tensorflow.keras import Input
import json
from sklearn.preprocessing import Normalizer


class BlackBox:

    image_shape = (128, 128, 3)

    def __init__(self, average_embedding_filepath):
        """Sorry for these redundant flags. When I wrote the class I tested it on images, therefore
        the conversion to embedding space was needed. However, during grouping alg. implementation I decided to
        use embeddings everywhere, that's why functions were adjusted with ifs. When I realized that this idea
        was bad, it was already too late to change anything :D 

        Args:
            average_embedding_filepath (Path, str): path to average embeddings of known classes.
        """
        self._load_model()
        self._load_average_embeddings(average_embedding_filepath)

    def _load_model(self):
        new_input = Input(shape=self.image_shape)
        model = ResNet50(include_top=False, weights="imagenet", input_tensor=new_input, pooling="avg")
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        self.model = model

    def _load_average_embeddings(self, path_to_embeddings):
        with open(path_to_embeddings, "r") as f:
            average_embeddings = json.load(f)

        parsed_average_embeddings = {cl: list(map(float, average_embeddings[cl])) for cl in average_embeddings.keys()}
        self.average_embeddings = parsed_average_embeddings
        embeddings = np.array(list(parsed_average_embeddings.values()))
        norm = Normalizer().fit(embeddings)
        self.normalizer = norm
        self.average_embeddings_values = norm.transform(embeddings)
        self.embeddings_classes = np.array(list(parsed_average_embeddings.keys()))
    
    def get_processed_image_embeddings(self, image, normalize=False):
        prediction = self.model.predict(image)
        prediction = self._normalize_vector(prediction) if normalize else prediction
        return prediction.reshape(prediction.shape[1])

    def _get_image_embeddings(self, image, normalize=False):
        processed_image = preprocess_input(image.reshape((1,) + self.image_shape))
        return self.get_processed_image_embeddings(processed_image, normalize=normalize)


    def _normalize_vector(self, vector):
        return self.normalizer.transform(np.array(vector).reshape(1, -1))

    def _calculate_distance(self, vector1, vector2, normalize=False):
        v1 = self._normalize_vector(vector1).flatten() if not normalize else vector1
        v2 = self._normalize_vector(vector2).flatten() if not normalize else vector2
        return np.linalg.norm(v1 - v2)

    def find_distances_between_image_and_known_classes(self, image, normalize=True, is_embedding=False):
        embedding = self._get_image_embeddings(image, normalize=normalize) if not is_embedding else image
        avg_distance_fun = lambda image: self._calculate_distance(image, embedding)
        avg_distances = list(map(avg_distance_fun, self.average_embeddings_values))
        return np.array(avg_distances)

    def find_closest_class(self, image, normalize=True, is_embedding=False):
        avg_distances = self.find_distances_between_image_and_known_classes(image, normalize=normalize, is_embedding=is_embedding)
        closest_embed = np.argmax(avg_distances)
        cl = list(self.embeddings_classes)[closest_embed]
        return cl

    def find_image_similarity(self, image1, image2, is_embedding=False, normalize=False):
        """The lower the output the more similar images are."""
        embedding1 = self._get_image_embeddings(image1) if not is_embedding else image1
        embedding2 = self._get_image_embeddings(image2) if not is_embedding else image2
        return self._calculate_distance(embedding1, embedding2, normalize=normalize)
