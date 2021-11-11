'''
Wikipedia image scraper, given a starting page on wikipedia this
script will scrape all the images and random choose the next wiki link
to scrape from, continue for given amount of pages
'''


#import libraries
from bs4 import BeautifulSoup, SoupStrainer
import requests
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint
import os


def get_images(url, page_no, print_image=False):
    '''
    :param url: (string) url of web page to scrape
    :param page_no: (int) used of storing images
    :param print_image: (boolean) whether to plot images or not
    '''
    html_page=requests.get(url) # get response from http request
    soup=BeautifulSoup(html_page.content, 'html.parser')
    images=soup.find_all('img')
    storage_directory="images/page"+str(page_no)
    try:
        os.mkdir(storage_directory)
    except FileExistsError:
        pass
    for i, image in enumerate(images):
        image_location=storage_directory+"/image"+str(i+1)+".jpg"
        url_ext = image.attrs['src']
        full_url = 'https:'+url_ext
        try:
            r = requests.get(full_url, stream=True)
        except:
            try:
                r = requests.get(url+url_ext, stream=True)
            except:
                continue

        if r.status_code==200:
            with open(image_location, "wb") as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
        if print_image:
            img = mpimg.imread(image_location)
            imgplot = plt.imshow(img)
            plt.show()


def next_page(current_page):
    '''
    :param current_page: url of page script is currently on
    :return next_url: url of page to scrape next
    '''
    html_page = requests.get(current_page)  # get response from http request
    links = BeautifulSoup(html_page.content, 'html.parser', parse_only=SoupStrainer('a'))
    links_list = []
    for link in links:
        if keep(link):
            links_list.append(link.get('href'))
    rand = get_random_number(len(links_list))
    next_url_ext=links_list[rand]
    next_url="https://en.wikipedia.org/"+str(next_url_ext)
    print("NEXT URL:" + next_url)

    return next_url


def keep(link):
    link_str = str(link)
    if not link.has_attr('href'):
        return False
    if "wiki" not in link_str:
        return False
    if "jpg" in link_str:
        return False
    if "wikipedia" in link_str:
        return False
    return True

def get_random_number(max_number):
    return randint(0, max_number-1)

def main(start_url, number_pages):
    '''
    :param start_url: url of the webpage to start scrapping from
    :param number_pages: how many pages to
    '''
    url = start_url
    for page in range(number_pages):
        get_images(url, page+1)
        url = next_page(url)


if __name__ == '__main__':
    main('https://en.wikipedia.org/wiki/', 7)