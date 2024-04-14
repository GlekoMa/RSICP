# Date: 2024.2
#
# Crawl all the calligraphy and painting images (include page HTML)
# from the National Palace Museum (Taipei).
#
# To run this script, you need
#   1. Selenium configured with the Chrome WebDriver.
#   2. Manually create:
#     - A directory to save images;
#     - A directory to save HTML;
#     - A csv file to save mapping information of images to HTML files.
#   3. Edit the `if __name__ == "__main__":` block according to your situation.

import sys
import os
import time
import csv
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


class NPMCrawler:
    """
    A Crawler to crawl all the calligraphy and painting images
    (include page HTML) from the National Palace Museum (Taipei).
    """

    def __init__(self, htmls_dir, images_dir, map_csv_path):
        self.htmls_dir = htmls_dir
        self.images_dir = images_dir
        self.map_csv_path = map_csv_path
        self.driver = self.init_chrome_driver()
        self.max_info_pid = self.get_max_info_pid()

    def init_chrome_driver(self, headless=False):
        """
        Init Chrome driver
        ---
        headless: bool, deciding whether to run Chrome in headless mode.
        """
        chrome_options = Options()
        if headless is True:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_experimental_option(
            "prefs",
            {"download.default_directory": os.path.join(os.getcwd(), self.images_dir)},
        )
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(0.5)
        return driver

    def get_max_info_pid(self):
        """
        Get existed max pid in 'map.csv'.
        """
        with open(self.map_csv_path, "r") as f:
            info_pids = [int(row[0]) for row in csv.reader(f)]
        return max(info_pids) if bool(info_pids) else 0

    def get_existing_imgs(self, filter_cr=True):
        """
        Get existed images (excluding currend downloading image).
        ---
        cr: bool, deciding filter crdownload or not.
        """
        imgs_dl = set(os.listdir(self.images_dir))
        if filter_cr is True:
            return set(filter(lambda x: x.split(".")[-1] != "crdownload", imgs_dl))
        else:
            return imgs_dl

    def fix(self):
        """
        If you break manually at the last run, chrome will
        ask you select continue download or end download.
        As unknown your choice, run this method will fix
        map.csv and your images/ by comparing
        existing images.
        """
        imgs_dl = self.get_existing_imgs()

        # delete the .crdownload file in images/
        # note: in most cases, Chrome auto deletes '.crdownload'
        # files, this is for handling exceptional situations.
        imgs_cr = self.get_existing_imgs(filter_cr=False) - imgs_dl
        if len(imgs_cr) == 0:
            pass
        elif len(imgs_cr) == 1:
            img_cr = imgs_cr.pop()
            os.remove(self.images_dir + "/" + img_cr)
        else:
            print("More than one '.crdownload' file, please check manually.")

        # add imgs downloaded last time which not yet added to map.csv
        if os.path.getsize(self.map_csv_path) == 0:
            return True  # quit if empty
        with open(self.map_csv_path, "r") as f:
            rows = list(csv.reader(f))
        imgs_info = set([row[1] for row in rows])
        next_pid = [int(row[0]) for row in rows][-1] + 1
        with open(self.map_csv_path, "a") as f:
            extra = imgs_dl - imgs_info
            if bool(extra):
                f.write(f"{next_pid},{next(iter(extra))}\n")
                self.max_info_pid = next_pid

    def get_img_dpi(self):
        """
        Note: need your driver loaded a page already.

        Check download bottons of the html, there are three situations:
            1. 100 DPI only;
            2. 100 DPI, 600 DPI both;
            3. no download botton.

        This function return 100, 600 (if 100, 600 both) or none.
        """
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        try:
            a = soup.find("a", id="a_600picture").attrs["style"]
            # lack 600 DPI download botton. s.g. pid 14349
            if "display: none;" in a:
                return 100
            else:
                return 600
        # also lack 100 DPI download botton. s.g. pid 36000
        except AttributeError:
            return None

    def download(self, pid, dpi):
        """
        Save page source (html) first, then download image,
        finally write downloaded image info to 'map.csv'.
        """
        with open(f"{self.htmls_dir}/{pid}.html", "w") as f:
            f.write(self.driver.page_source)

        # get the ID value for a given DPI
        if dpi == 100:
            id_v = "a_download"
        elif dpi == 600:
            id_v = "a_600picture"
        else:
            with open(self.map_csv_path, "a") as f:
                f.write(f"{pid},None\n")
            return

        # download image
        dl_botton = self.driver.find_element(by=By.ID, value=id_v)
        imgs_before_dl = self.get_existing_imgs()
        dl_botton.click()

        # wait for download completely
        while self.get_existing_imgs() == imgs_before_dl:
            time.sleep(1)

        # write to 'map.csv'
        img_new = next(iter(self.get_existing_imgs() - imgs_before_dl))
        with open(self.map_csv_path, "a") as f:
            f.write(f"{pid},{img_new}\n")
        return

    def run(self):
        """
        Check your 'map.csv' and fix it, then begin to save
        related page HTML and crawl all images (previously saved
        will not be crawled.)
        """
        self.fix()
        for pid in range(self.max_info_pid + 1, 36599):
            url = f"https://digitalarchive.npm.gov.tw/Painting/Content?pid={pid}&Dept=P"
            self.driver.get(url)
            self.download(pid, dpi=self.get_img_dpi())
            time.sleep(1)
        print("Completed!")
        self.driver.quit()


def is_normal(images_dir, map_csv_path):
    """
    check if existed images and mapping info are right.
    """
    # Return True if dl_info.csv is empty
    if os.path.getsize(map_csv_path) == 0:
        return True

    imgs_dl = set(os.listdir(images_dir))

    with open(map_csv_path, "r") as f:
        reader = csv.reader(f)
        pids_info, imgs_info = zip(*[(row[0], row[1]) for row in reader])
    imgs_info_exist = set(filter(lambda x: x != "None", imgs_info))

    # 1. check pids is continuous
    pids_info = list(int(i) for i in pids_info)
    pids_need = list(range(1, len(pids_info) + 1))
    if pids_info != pids_need:
        print(f"Error: pids in {map_csv_path} is not continuous")
        print("Note:")
        print("\tpids_need - pids_info:", set(pids_need) - set(pids_info))
        return False
    # 2. check imgs_dl == imgs_info_exist
    elif imgs_dl != imgs_info_exist:
        cond_1 = len(imgs_info_exist - imgs_dl) == 0
        cond_2 = len(imgs_dl - imgs_info_exist) == 1
        # This situation have handled in main.py (NPMCrawler.fix)
        if cond_1 and cond_2:
            if __name__ == "__main__":
                print("No error, good!")
            return True
        else:
            print("Error: images is not equal to dl_info.csv")
            print("Note:")
            print(f"\tquantity {len(imgs_info_exist)} VS quantity {len(imgs_dl)}")
            print("\timgs_info_exist - imgs_dl:", imgs_info_exist - imgs_dl)
            print("\timgs_dl - imgs_info_exist:", imgs_dl - imgs_info_exist)
            return False
    else:
        if __name__ == "__main__":
            print("No error, good!")
        return True


if __name__ == "__main__":
    htmls_dir = "../data/Chinese-Painting/htmls"
    images_dir = "../data/Chinese-Painting/images"
    map_csv_path = "../data/Chinese-Painting/map.csv"
    err = "does't exist, please create it or specify another."
    for path in [htmls_dir, images_dir, map_csv_path]:
        if not os.path.exists(path):
            print(f"Error: {path} {err}")
            sys.exit()
    if is_normal(images_dir, map_csv_path) is True:
        crawler = NPMCrawler(htmls_dir, images_dir, map_csv_path)
        crawler.run()
    else:
        sys.exit(1)
