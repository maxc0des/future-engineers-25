#code to prepare the data befor training it to the network
import pandas as pd
import cv2
import random

#================================
#change the parameters here:

data_path = "path/to/image"
target_path = "path/to/save/adjusted_data.csv"

data_filters = [["steering", 80, ">"]]

picture_adjustments = [["brightness", 23, 10]]  # feature, value < 255 (max), iterations

#===================================

def filter_data(feature: str, value: int, comparison: chr, df: pd.DataFrame):
    if comparison == ">":
        df = df[df[feature] > value]
    elif comparison == "<":
        df = df[df[feature] < value]
    elif comparison == "=":
        df = df[df[feature] == value]
    elif comparison == "!":
        df = df[df[feature] != value]
    else:
        raise ValueError("Comparison not recognized")
    return df

def adjust_picture(feature: str, value: int, img_path: str):
    image = cv2.imread(img_path)
    if feature == "brightness":
        if value:
            image = cv2.add(image, value)
        else:
            value = value * -1
            image = cv2.subtract(image, value)
        filename = img_path + "_adjusted"
        cv2.imwrite(filename, image)
        return filename
    else:
        raise ValueError("Feature not recognized for adjustment.")
        
if __name__ == "__main__":
    input("make sure to backup the df before running this")
    df = pd.read_csv(data_path)
    df_length = df.shape[0]
    for filter in data_filters:
        feature, value, comparison = filter
        df = filter_data(feature, value, comparison, df)
    print("Finished filtering the data, starting to adjust the pictures.")
    for adjustment in picture_adjustments:
        feature, value, iterations = adjustment
        for i in range(iterations):
            value = random.randint(value*-1, value)
            random_index = random.randint(0, df_length - 1)
            img_path = df.loc[random_index, 'cam_path']
            new_img_path = adjust_picture(feature, value, img_path)
            new_row = pd.DataFrame({'cam_path': [new_img_path]})
            df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(target_path, index=False)
    print(f"Data has been saved to {target_path}.")