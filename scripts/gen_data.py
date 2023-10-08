from src.data.components.gen_data import ChessDataGenerator

if __name__ == "__main__":
    input_path = "./data/20230929_raw_data"
    train_cases_path = "./data/train_cases.npz"
    val_cases_path = "./data/val_cases.npz"
    # ChessDataGenerator().convert_data(input_path, train_cases_path, val_cases_path)
    # ChessDataGenerator().convert_data_from_realworld(input_path, train_cases_path, val_cases_path)
    ChessDataGenerator(eval_option = True).generate_data(500, "./data/train_1000_gen_cases.npz")
