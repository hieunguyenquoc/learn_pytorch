import pandas as pd

class create_data:
    @staticmethod
    def create_train_data():
        with open("data_text/train.en", "r", encoding="utf-8") as f_en:
            open_en = f_en.readlines()

        with open("data_text/train.vi", "r", encoding="utf-8") as f_vi:
            open_vi = f_vi.readlines()
        
        #create dataframe    
        data = {'id': range(1, len(open_vi) + 1), 'en': open_en, 'vi': open_vi}
        df = pd.DataFrame(data)

        df.to_csv('data_csv/train.csv', index=False, encoding='utf-8')

        print("Tạo thành công file train")

    @staticmethod
    def create_test_data():
        with open("data_text/test.en", "r", encoding="utf-8") as f_en:
            open_en = f_en.readlines()

        with open("data_text/test.vi", "r", encoding="utf-8") as f_vi:
            open_vi = f_vi.readlines()
        
        #create dataframe    
        data = {'id': range(1, len(open_vi) + 1), 'en': open_en, 'vi': open_vi}
        df = pd.DataFrame(data)

        df.to_csv('data_csv/test.csv', index=False, encoding='utf-8')

        print("Tạo thành công file test")

    @staticmethod
    def create_dev_data():
        with open("data_text/dev.en", "r", encoding="utf-8") as f_en:
            open_en = f_en.readlines()

        with open("data_text/dev.vi", "r", encoding="utf-8") as f_vi:
            open_vi = f_vi.readlines()
        
        #create dataframe    
        data = {'id': range(1, len(open_vi) + 1), 'en': open_en, 'vi': open_vi}
        df = pd.DataFrame(data)

        df.to_csv('data_csv/dev.csv', index=False, encoding='utf-8')

        print("Tạo thành công file dev")

if __name__ == "__main__":
    data = create_data()
    data.create_test_data()
    data.create_dev_data()
