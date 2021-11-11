from bs4 import BeautifulSoup
import requests


def get_sonnets(url):
    html_page = requests.get(url)  # get response from http request
    soup = BeautifulSoup(html_page.content, 'html.parser')
    sonnets = soup.find_all(text=True)
    cut_off_front, cut_off_end = 0, 0

    for index, sentence in enumerate(sonnets):
        if sentence == 'I.':
            cut_off_front = index
        if sentence == ' Sonnetads2 ':
            cut_off_end = index

    sonnets = sonnets[cut_off_front:cut_off_end]

    file_all = open("sonnets/all_sonnets.txt", "w")
    file_all.writelines(sonnets)



def main(url):
    get_sonnets(url)


if __name__ == '__main__':
    main('http://www.shakespeares-sonnets.com/all.php')