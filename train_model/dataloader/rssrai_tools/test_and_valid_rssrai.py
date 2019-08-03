from sklearn.model_selection import train_test_split
import pandas as pd
def fun():
    df = pd.read_csv("valid_label.csv")
    train_set, valid_set = train_test_split(df,test_size=0.1, random_state=123)
    train_set.to_csv("train_set.csv",index=0)
    valid_set.to_csv("valid_set.csv",index=0)

if __name__ == '__main__':
    fun()
