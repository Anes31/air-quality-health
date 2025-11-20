from src.train_risk_model import main as train_main


def main():
    print("=== Running daily training (train_risk_model) ===")
    train_main()
    print("=== Daily training finished ===")


if __name__ == "__main__":
    main()