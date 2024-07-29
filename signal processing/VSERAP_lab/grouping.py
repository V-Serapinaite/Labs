"""Image grouping algorithm."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from random import sample
from itertools import chain
import seaborn as sn

from blackbox import BlackBox
from typing import Union


class GroupingModel:

    big_number = 1000

    def __init__(self, average_embedding_filepath: Union[str, Path]):
        self.black_box = BlackBox(
            average_embedding_filepath=average_embedding_filepath
        )  # initializes black box model. Treat is as a black box.
        self.continue_grouping = True
        self.iterations = 0

    def fit(
        self,
        images: list,
        similarity_threshold_for_seen_classes: float = 0.5,
        similarity_threshold_for_unseen_classes: float = 0.5,
        maximum_number_of_created_classes: Union[int, None] = None,
        max_iterations: Union[int, None] = 5,
        similarity_increment_step: Union[int, None] = 0.05,
    ) -> None:
        """Create groups of images based on their embeddings.

        The algorithm contains average embeddings information of 15 different classes (categories). They can be found
            in avg_embeddings.txt file. These images are called `seen`.
        Algorithm steps:
        1. Find all image embeddings (vector representations of image features, such as patterns).
        2. Find which image embeddings were `seen` and which are not. Since the algorithm contains average embeddings for each
            `seen` class, we can find to which class new embeddings belong to.
        3. Iteratively create groups of embeddings. In the beginning, each embedding is considered as separate group.
            During the iterations, groups which are the closest to each other are joined, and new average embedding is
            calculated. In this way the number of groups gets reduced till finally all created groups are not similar
            enough based on the threshold.
        4. Minimize the number of groups till it reaches maximum number of created groups. Grouping threshold is
            increased and 3rd step is repeated till group number is reduced.


        Args:
            images (list): List of images used for grouping.
            similarity_threshold_for_seen_classes (float, optional): Threshold of similarity (distance) for `seen` classes. Defaults to 0.5.
            similarity_threshold_for_unseen_classes (float, optional): Threshold of similarity (distance) for `seen` classes. Defaults to 0.5.
            maximum_number_of_created_classes (Union[int, None], optional): he maximum number of groups algorithm can create. If
                algorithm created more classes, they will be joined until they reach this number. If None, then there
                is no such limitation. Defaults to None.
            max_iterations (Union[int, None], optional): Number of max iterations for grouping process.
            similarity_increment_step (Union[int, None], optional): Determines how much distance thresholds should be
                incremented during each iteration for 4. step. Defaults to 0.05.

        """
        if maximum_number_of_created_classes is not None:
            assert (
                maximum_number_of_created_classes >= 1
            ), "maximum_number_of_created_classes param has to be at least 1!"

        images = self.process_images(images)
        self.max_iterations = max_iterations
        self.maximum_number_of_created_classes = maximum_number_of_created_classes
        self.similarity_threshold_for_seen_classes = similarity_threshold_for_seen_classes
        self.similarity_threshold_for_unseen_classes = similarity_threshold_for_unseen_classes
        self.similarity_increment_step = similarity_increment_step
        self.find_seen_images(images)
        self.perform_grouping_of_unseen_classes(self.unseen_images)
        self.fitted = True

    def fit_predict(self, images, **kwargs):
        """Predicts labels for the images on which model was fitted."""
        if not self.fitted:
            self.fit(images, **kwargs)
        # get predictions here
        final_predictions = np.zeros(len(images), dtype=object)
        final_predictions[self.seen_images_indexes] = self.seen_images_classes

        temp_predictions = self.inner_predict(self.unseen_images)
        final_predictions[self.unseen_images_indexes] = temp_predictions
        return final_predictions

    def process_images(self, images):
        """Find embeddings of each image."""

        def processing_fun(img):
            return self.black_box._get_image_embeddings(img, normalize=True)

        processed_image_list = list(map(processing_fun, images))
        return processed_image_list

    def find_seen_images(self, images):
        """Find images which are `seen` by model."""
        find_distances_fun = lambda img: self.black_box.find_distances_between_image_and_known_classes(
            img, is_embedding=True
        )
        similarities = np.array(list(map(find_distances_fun, images)))
        find_if_have_similarities = np.where(
            similarities <= self.similarity_threshold_for_seen_classes, similarities, 0
        )
        have_similarities = np.nonzero(find_if_have_similarities)
        # finds images which are seen
        seen_images_indexes = np.unique(have_similarities[0])
        self.seen_images = np.array(images)[seen_images_indexes]
        self.seen_images_indexes = seen_images_indexes

        unseen_images_indexes = ~np.any(find_if_have_similarities, axis=1)
        self.unseen_images = np.array(images)[unseen_images_indexes]
        self.unseen_images_indexes = unseen_images_indexes

        # finds seen image class: if image is similar to multiple classes, we find a class with which it is the most similar
        temp_found_images = find_if_have_similarities[seen_images_indexes, :]
        images_with_known_classes = np.where(
            temp_found_images > 0, temp_found_images, self.similarity_threshold_for_seen_classes + 1
        )
        known_class_indexes = np.argmin(images_with_known_classes, axis=1)
        self.seen_images_classes = self.black_box.embeddings_classes[known_class_indexes]

    def group_unseen_images(self, images):
        """Create groups of unseen images, see 3rd step."""
        if not len(images):
            return
        distances = np.array(
            [
                np.array(
                    [
                        self.black_box.find_image_similarity(img1, img, is_embedding=True)
                        if ind1 != ind
                        else self.big_number
                        for ind1, img1 in enumerate(images)
                    ]
                )
                for ind, img in enumerate(images)
            ]
        )

        find_if_have_similarities = np.where(distances <= self.similarity_threshold_for_unseen_classes, distances, 0)
        have_similarities = np.nonzero(find_if_have_similarities)
        similar_image_indexes = np.unique(have_similarities[0])
        if not np.any(similar_image_indexes):
            self.continue_grouping = False
            return

        self.iterations += 1
        distances_df = pd.DataFrame(distances)
        filtered_distances_df = distances_df[distances_df.index.isin(similar_image_indexes)]
        not_similar_images = distances_df[~distances_df.index.isin(similar_image_indexes)].index
        most_similar = filtered_distances_df.idxmin(axis=1)
        most_similar_rows, most_similar_cols = most_similar.index, most_similar.values
        similar_row_images, similar_col_images = images[most_similar_rows], images[most_similar_cols]
        new_embeddings = self._combining_embeddings(similar_row_images, similar_col_images)

        adjusted_images = list(images[not_similar_images].copy())
        adjusted_images.extend(np.unique(new_embeddings, axis=0))
        return np.array(adjusted_images)

    def _combining_embeddings(self, similar_row_images, similar_col_images):
        """Combines 2 embeddings into 1."""
        new_embeddings = []  # TODO: add some weights for embeddings
        for emb1, emb2 in zip(similar_row_images, similar_col_images):
            new_emb = (emb1 + emb2) / 2
            new_embeddings.append(new_emb)
        return new_embeddings

    def perform_grouping_of_unseen_classes(self, images):
        """Iteratively groups image embeddings, see step 3."""
        used_images = images
        while (
            self.max_iterations >= self.iterations if self.max_iterations is not None else True
        ) and self.continue_grouping:
            imgs = self.group_unseen_images(used_images)
            if imgs is not None:
                used_images = imgs

        if self.maximum_number_of_created_classes is not None:
            self.similarity_threshold_for_unseen_classes += (
                self.similarity_increment_step
            )  # TODO: add maximum similarity threshold
            while len(used_images) > self.maximum_number_of_created_classes:
                imgs = self.group_unseen_images(used_images)
                if imgs is not None:
                    used_images = imgs

        self.centroids = used_images

    def inner_predict(self, images):
        """Find which group embedding is closest to each embeddings."""
        distances = np.array(
            [
                np.array(
                    [
                        self.black_box.find_image_similarity(img1, img, is_embedding=True)
                        if ind1 != ind
                        else self.big_number
                        for ind1, img1 in enumerate(self.centroids)
                    ]
                )
                for ind, img in enumerate(images)
            ]
        )
        df = pd.DataFrame(distances)

        # since the name of the groups does not have any meaning, we take group column index values as class names
        predictions = np.array(df.idxmin(axis=1).values, dtype=object)
        return predictions


def load_images(
    image_folder: Path,
    unused_images_df: pd.DataFrame,
    load_unseen_classes: int = 15,
    load_images_per_class: int = 10,
    load_seen_classes: int = 5,
) -> Union[list, np.array]:
    """Loads `seen` and `unseen` images for model fitting and testing."""

    def load_images_inner(df, class_name, load_images_per_class=10):
        df_filt = df[df["class"] == class_name]
        indexes = sample(list(df_filt.index), load_images_per_class)
        paths = df_filt.loc[indexes, ["image_full_path"]].values.flatten()
        images = [np.array(Image.open(p)) for p in paths]
        labels = df_filt.loc[indexes, ["class"]]
        return np.array(images), labels.values

    images_paths = list(image_folder.glob("*.png"))
    image_names = list(map(lambda x: x.name, images_paths))
    new_df = pd.DataFrame(images_paths, columns=["image_full_path"])
    new_df["image_name"] = image_names
    new_df["extra"] = new_df.image_name.apply(lambda x: x[:-4].split("__"))
    new_df[["class", "image_number"]] = new_df["extra"].apply(pd.Series)
    new_df.drop(columns="extra", axis=1, inplace=True)
    unseen_image_df = new_df[~new_df["class"].isin(unused_images_df["class"].unique())]
    unseen_image_classes = unseen_image_df["class"].unique()
    selected_unseen_classes = sample(list(unseen_image_classes), load_unseen_classes)
    seen_image_df = unused_images_df.copy()
    seen_image_df["image_full_path"] = unused_images_df["image_name"].apply(lambda x: image_folder / x)
    selected_seen_classes = sample(list(seen_image_df["class"].unique()), load_seen_classes)
    testing_set = []
    testing_labels = []

    max_samples_in_unseen_images = len(unseen_image_df[unseen_image_df["class"] == selected_unseen_classes[0]])

    def load_unseen_images_fun(cl):
        return load_images_inner(
            unseen_image_df,
            cl,
            load_images_per_class=load_images_per_class
            if load_images_per_class <= max_samples_in_unseen_images
            else max_samples_in_unseen_images,
        )

    images_and_labels_unseen = list(map(load_unseen_images_fun, selected_unseen_classes))
    unseen_images = list(chain(*list(zip(*images_and_labels_unseen))[0]))
    unseen_labels = list(chain(*list(zip(*images_and_labels_unseen))[1]))
    testing_set.extend(unseen_images)
    testing_labels.extend(unseen_labels)

    max_samples_in_seen_images = len(seen_image_df[seen_image_df["class"] == selected_seen_classes[0]])

    def load_seen_images_fun(cl):
        return load_images_inner(
            seen_image_df,
            cl,
            load_images_per_class=load_images_per_class
            if load_images_per_class <= max_samples_in_seen_images
            else max_samples_in_unseen_images,
        )

    images_and_labels_seen = list(map(load_seen_images_fun, selected_seen_classes))
    seen_images = list(chain(*list(zip(*images_and_labels_seen))[0]))
    seen_labels = list(chain(*list(zip(*images_and_labels_seen))[1]))
    testing_set.extend(seen_images)
    testing_labels.extend(seen_labels)
    return testing_set, list(chain(*testing_labels))


def plot_random_images(images, show_n_images, subtitles=None):
    """Plots some random images from images."""
    fig, ax = plt.subplots(show_n_images, figsize=(20, 20))
    plt_images = sample(list(images), show_n_images)
    for i in range(show_n_images):
        ax[i].imshow(plt_images[i])
        if subtitles is not None:
            ax[i].set_title(subtitles[i])
    plt.show()


def calculate_purity(res_df):
    """Calculates purity (homogeneity) in formed groups."""
    clusters = res_df["prediction"].unique()
    counts = 0
    for cl in clusters:
        flt_gr = res_df[res_df["prediction"] == cl]
        gr_count = flt_gr["labels"].value_counts()[0]
        counts += gr_count
    return counts / len(res_df)


if __name__ == "__main__":
    """Download https://www.kaggle.com/jessicali9530/coil100 dataset."""

    average_embedding_filepath = Path("avg_embeddings.txt")
    unused_images_path = Path("unused_images.csv")
    image_folder = Path("coil-100")
    unused_images_df = pd.read_csv(unused_images_path)

    testing_images, testing_labels = load_images(
        image_folder, unused_images_df, load_unseen_classes=15, load_images_per_class=10, load_seen_classes=5
    )

    plot_random_images(testing_images, 5)
    grouping_model = GroupingModel(average_embedding_filepath)
    seen_images_class_names = grouping_model.black_box.embeddings_classes

    # Feel free to modify these values. Don't try to break the model because it will break.
    grouping_model.fit(
        testing_images, maximum_number_of_created_classes=30, similarity_increment_step=0.5
    )

    # get labels of images on which model was fitted.
    preds = grouping_model.fit_predict(testing_images)

    res_df = pd.DataFrame.from_dict({"prediction": preds, "labels": testing_labels})
    res_df["was_seen"] = res_df["labels"].apply(lambda x: x in seen_images_class_names)
    # prediction vs real label contigency table.
    contingency_table = pd.crosstab(index=res_df["labels"], columns=res_df["prediction"])

    fig, ax = plt.subplots(1, figsize=(15, 10))
    sn.heatmap(contingency_table, ax=ax)

    purity = calculate_purity(res_df)
    print(f"Purity is: {purity}")
