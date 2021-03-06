import os
import pandas as pd

from datasets.abstract_DR import Abstract_DR_Dataset

class EyePACS_Dataset(Abstract_DR_Dataset):
    def __init__(self, data_directory, img_size=384, use_inception_norm=True, random_state=None, max_length=None, remove_ungradables=True):
        # Dataset specific file locations
        self.labels_fname = os.path.join(data_directory, "trainLabels.csv", "trainLabels.csv")
        self.gradability_fname = os.path.join(data_directory, "eyepacs_gradability_grades.csv")
        # Init dataset
        super().__init__(data_directory, img_size, use_inception_norm, max_length, random_state=random_state, remove_ungradables=remove_ungradables)

    def load_labels(self, max_length, **kwargs):
        # Load label csv to dataframe
        labels_df = pd.read_csv(self.labels_fname)
        if kwargs["remove_ungradables"]:
            labels_df = self.remove_ungradables(labels_df)
        labels_df = self.labels_to_binary(labels_df)
        # Random shuffle
        labels_df = labels_df.sample(frac=1, random_state=kwargs["random_state"]).reset_index(drop=True)
        # Choose number of samples to keep
        labels_df = labels_df.iloc[:max_length] if max_length is not None else labels_df
        # Standard column names between classes
        labels_df = labels_df.rename(columns={"image": 'image_name'})
        labels_df["image_name"] = [fname_img + ".jpeg" for fname_img in labels_df["image_name"]]
        return labels_df

    def remove_ungradables(self, labels_df):
        # Load gradabilty csv to dataframe
        gradabilty_df = pd.read_csv(self.gradability_fname, delimiter = " ")
        # Merge label and gradabilty dataframes
        labels_df = pd.merge(labels_df, gradabilty_df, how="inner", left_on=labels_df.columns[0], right_on=gradabilty_df.columns[0])
        # Drop ungradables and non-required columns
        labels_df = labels_df.drop(labels_df[labels_df["gradability"]==0].index)
        labels_df = labels_df.drop(columns=['image_name', 'gradability'])
        return labels_df
