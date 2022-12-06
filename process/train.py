import pandas as pd
from prophet import Prophet
from ops import save_model, make_prediction

INPUT_FILE = "data/train.csv"


if __name__ == '__main__':
    # read training data
    df = pd.read_csv(INPUT_FILE)

    # create and train model
    m = Prophet()
    m.fit(df)

    # save model weights
    save_model(model=m, save_path="weights/model_weights.json")
    # make prediction
    make_prediction(model=m, file_name="data/prediction.csv")
