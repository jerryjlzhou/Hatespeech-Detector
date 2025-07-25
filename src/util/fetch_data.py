import csv
import requests
from pathlib import Path
from bs4 import BeautifulSoup


def extract_racial_slur_database(
            output_path: str,
            delimiter: str = '\t'
        ):

    url = "http://www.rsdb.org/full"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=delimiter)

        table = soup.find("table")
        for row in table.find_all("tr"):
            cols = row.find_all(["td", "th"])
            data = [col.get_text(strip=True) for col in cols]
            if data:
                writer.writerow(data)


def main():
    BASE_DIR = Path(__file__).resolve().parent
    output_path = BASE_DIR / 'racial_slur_database.csv'
    extract_racial_slur_database(output_path)


if __name__ == '__main__':
    main()
