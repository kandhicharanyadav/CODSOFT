from datasets import load_dataset

def main():
    try:
        ds = load_dataset("dwzhu/PaperBananaBench")
        print(ds)
        # Check what features exist in the train split
        if 'train' in ds:
            print("\nTrain features:")
            print(ds['train'].features)
            print("\nFirst sample:")
            print(ds['train'][0])
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main()
