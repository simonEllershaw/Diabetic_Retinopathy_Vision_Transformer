import os
import pandas as pd

from datasets.abstract_DR import Abstract_DR_Dataset


class Messidor_Dataset(Abstract_DR_Dataset):
    def __init__(self, data_directory, img_size=384, use_inception_norm=True, labels_to_binary=True, max_length=None):
        super().__init__(data_directory, img_size, use_inception_norm, max_length, labels_to_binary=labels_to_binary)

    def load_labels(self, max_length, **kwargs):
            labels_df = self.load_labels_from_sub_dirs()
            # Standard column names between classes
            labels_df = labels_df.rename(columns={"Image name": 'image_name', "Retinopathy grade": 'level'})
            labels_df = labels_df.reset_index()
            labels_df = self.fix_erratas(labels_df)
            if kwargs["labels_to_binary"]:
                labels_df = self.labels_to_binary(labels_df)
            labels_df = labels_df.iloc[:max_length] if max_length is not None else labels_df
            labels_df = labels_df.reset_index(drop=True)
            return labels_df
    
    def load_labels_from_sub_dirs(self):
        labels_df = pd.DataFrame()
        # File structure is a series of subdirectories
        for item in os.listdir(self.data_directory):
            if os.path.isdir(os.path.join(self.data_directory, item)):
                sub_dir = item
                sub_dir_full_path = os.path.join(self.data_directory, sub_dir)
                # Each sub_dir has an .xls with labels of images in that sub_dir
                # Open these and add to labels_df
                for fname in os.listdir(sub_dir_full_path):
                    if fname.endswith(".xls"):
                        annotations_fname = os.path.join(sub_dir_full_path, fname)
                        base_labels_df = pd.read_excel(annotations_fname)
                        base_labels_df["Image name"] = [os.path.join(sub_dir, fname_img) for fname_img in base_labels_df["Image name"]]
                        labels_df = labels_df.append(base_labels_df)
        return labels_df
    
    def fix_erratas(self, labels_df):
        # Erratas defined https://www.adcis.net/en/third-party/messidor/
        # Correct incorrect labels
        labels_df.loc[labels_df["image_name"] == os.path.join("Base11", "20051020_64007_0100_PP.tif"), "Retinopathy grade"] = 3
        labels_df.loc[labels_df["image_name"] == os.path.join("Base11", "20051020_63936_0100_PP.tif"), "Retinopathy grade"] = 1
        labels_df.loc[labels_df["image_name"] == os.path.join("Base13", "20060523_48477_0100_PP.tif"), "Retinopathy grade"] = 3
        labels_df.loc[labels_df["image_name"] == os.path.join("Base11", "20051020_63045_0100_PP.tif"), "Retinopathy grade"] = 0
        # Remove duplicates
        base_33_duplicates = ["20051202_55582_0400_PP.tif", "20051202_41076_0400_PP.tif", "20051202_48287_0400_PP.tif", "20051202_48586_0400_PP.tif", "20051202_55457_0400_PP.tif", "20051202_55626_0400_PP.tif", "20051202_54783_0400_PP.tif", "20051202_48575_0400_PP.tif", "20051205_32966_0400_PP.tif", "20051202_55484_0400_PP.tif", "20051205_32981_0400_PP.tif", "20051202_55562_0400_PP.tif", "20051202_54547_0400_PP.tif"]
        for duplicate in base_33_duplicates:
            labels_df = labels_df.drop(labels_df[labels_df["image_name"] == os.path.join("Base33", duplicate)].index)        
        return labels_df
