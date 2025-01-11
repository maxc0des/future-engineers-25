import os
import pandas as pd
import cv2
import random

#================================
# Change the parameters here:
data_path = "sample_data.csv"
target_path = "samples"

data_filters = [["steering", 80, "<"]]
picture_adjustments = [["brightness", 200, 10]]  # feature, value < 255 (max), iterations
validation = 5
test = 10
#===================================

def filter_data(feature: str, value: int, comparison: str, df: pd.DataFrame) -> pd.DataFrame:
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

def adjust_picture(feature: str, value: int, img_path: str) -> str:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image path {img_path} does not exist.")
    image = cv2.imread(img_path)
    if feature == "brightness":
        adjustment = value if value >= 0 else -value
        if value > 0:
            image = cv2.add(image, adjustment)
        else:
            image = cv2.subtract(image, adjustment)
        filename = f"adjusted_{random.randint(1000, 9999)}_{os.path.basename(img_path)}"
        new_path = os.path.join(os.path.dirname(img_path), filename)
        cv2.imwrite(new_path, image)
        return new_path
    else:
        raise ValueError("Feature not recognized for adjustment.")

if __name__ == "__main__":
    input("Make sure to back up the DataFrame before running this.")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist.")
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    df = pd.read_csv(data_path)
    print(f"Loaded data with {df.shape[0]} rows.")
    for feature, value, comparison in data_filters:
        if feature not in df.columns:
            raise KeyError(f"Feature {feature} not found in DataFrame.")
        df = filter_data(feature, value, comparison, df)
    print("Finished filtering the data.")
    if df.empty:
        raise ValueError("The DataFrame is empty after applying filters. Please adjust the filtering criteria.")
    for feature, value, iterations in picture_adjustments:
        for i in range(iterations):
            random_index = random.randint(0, len(df) - 1)
            original_row = df.iloc[random_index].copy()
            img_path = original_row['cam_path']
            new_img_path = adjust_picture(feature, random.randint(-value, value), img_path)
            new_row = original_row.copy()
            new_row['cam_path'] = new_img_path
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    print("Finished adjusting pictures.")
    df = df.sample(frac=1).reset_index(drop=True)
    validation_df = df.iloc[::validation].reset_index(drop=True)
    test_df = df.iloc[::test].reset_index(drop=True)
    train_df = df.drop(validation_df.index).reset_index(drop=True)
    val_path = os.path.join(target_path, "validation_data.csv")
    train_path = os.path.join(target_path, "train_data.csv")
    test_path = os.path.join(target_path, "test_data.csv")
    validation_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    train_df.to_csv(train_path, index=False)
    print(f"Data has been saved to {target_path}.")