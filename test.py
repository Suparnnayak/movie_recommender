import pickle

with open("artifacts/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(type(metadata))  # Should be <class 'pandas.core.frame.DataFrame'>
