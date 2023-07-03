import pandas as pd


def xlsx_to_csv(input_file_path, output_file_path):
    # Load spreadsheet
    xl = pd.read_excel(input_file_path)

    # Save it as csv file
    xl.to_csv(output_file_path, index=False, encoding='utf-8')


# usage
# xlsx_to_csv('../Book1.xlsx', '../data/economic_data.csv')
