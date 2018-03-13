import numpy as np
import requests
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import urllib.parse
import datetime
from io import BytesIO
import PIL
from PIL import Image
import imageio
import json
import re
import hashlib
import os
import os.path
from multiprocessing import Pool

BASE_URL = 'https://maps.googleapis.com/maps/api/staticmap?'
BASE_URL_GEOCODE = 'https://maps.googleapis.com/maps/api/geocode/json?'
THUMBNAIL_DEFAULT_SIZE = int(64)


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def address_to_filename(address):
    address = address.replace('ä', 'ae')
    address = address.replace('ö', 'oe')
    address = address.replace('ü', 'ue')
    address = address.replace('Ä', 'Ae')
    address = address.replace('Ö', 'Oe')
    address = address.replace('Ü', 'Ue')
    address = address.replace('ß', 'ss')
    # Keep only basic english letters a-zA-Z, numbers and whitespaces
    address = re.sub(r'[^a-zA-Z0-9\s]+', '', address)
    return address


def create_thumbnail(data):
    from_file = data[0]
    to_file = data[1]
    size = data[2]
    image = Image.open(from_file)
    image = image.resize((size, size), resample=PIL.Image.LANCZOS)
    image.save(to_file)


def create_thumbnails(download_folder, thumbs_sub_folder='thumbs', thumbnail_size=THUMBNAIL_DEFAULT_SIZE,
                      skip_existing=True):
    thumbnail_counter = int(0)
    skip_counter = int(0)
    data = list()
    with os.scandir(download_folder) as it:
        thumbs_folder = os.path.join(download_folder, thumbs_sub_folder)
        create_folder_if_not_exists(thumbs_folder)
        for entry in it:
            if entry.is_file() and entry.name != '.DS_Store':
                thumbnail_path = os.path.join(thumbs_folder, entry.name)
                if skip_existing and os.path.isfile(thumbnail_path):
                    skip_counter += 1
                    continue

                data.append((entry.path, thumbnail_path, thumbnail_size))

                thumbnail_counter += 1

    with Pool(processes=8) as p:
        p.map(create_thumbnail, data)

    print("Total: {0} thumbnails of size {1}x{1} were created, {2} skipped".format(thumbnail_counter, thumbnail_size, skip_counter))


class DownloadSession:
    def __init__(self, api_key='', download_folder='images', thumbs_sub_folder='thumbs', image_size=300, crop_size=25,
                 thumbnail_size=THUMBNAIL_DEFAULT_SIZE, thumbnail_prefix=''):

        if (not thumbs_sub_folder) and (not thumbnail_prefix):
            raise ValueError('thumbnail_prefix and thumbnail_folder cannot both be empty')

        # Retry image download if Google throws a 500 internal error
        self.retries = Retry(total=3,
                             backoff_factor=0.5,
                             status_forcelist=[500, 502, 503, 504])

        self.api_key = api_key

        # Default values for API call
        self.maps_api_options = {
                'size': '{}x{}'.format(image_size, image_size+2*crop_size),
                'zoom': '20',
                'maptype': 'satellite',
                'format': 'png32',
                'key': api_key,
        }

        self.hash_sums = dict()
        self.hash_method = hashlib.md5

        self.download_folder = download_folder
        create_folder_if_not_exists(self.download_folder)
        self.thumbs_folder = os.path.join(self.download_folder, thumbs_sub_folder)
        create_folder_if_not_exists(self.thumbs_folder)

        self.image_size = image_size
        self.crop_size = crop_size
        self.thumbnail_size = thumbnail_size
        self.thumbnail_prefix = thumbnail_prefix

    def get_file_hash(self, filename):
        hash_value = self.hash_method()
        with open(filename, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_value.update(chunk)
        return hash_value.hexdigest()

    def get_hash(self, data):
        hash_value = self.hash_method()
        hash_value.update(data)
        return hash_value.hexdigest()

    def update_api_options(self, **kwargs):
        self.maps_api_options.update(kwargs)

    def build_google_api_url(self, **kwargs):
        """

        :param kwargs: see below

        :Keyword Arguments:
            * *center* (``string``) --
               A street address or GPS coordinates that will be passed to Google Maps

        :return: ``string`` Full URL to the Google Maps Static API call
        """

        options = self.maps_api_options.copy()
        options.update(kwargs)

        if 'center' not in options:
            raise TypeError("Missing required argument: 'center'")

        url = BASE_URL + urllib.parse.urlencode(options)

        return url

    def check_address_existence(self, address, street_number_used):
        """

        :param address: Full address, which will be geocoded using Google's API
        :param street_number_used: the street number part of address
        :return: True if the address exists on Google Maps, False otherwise
        """
        url = BASE_URL_GEOCODE + urllib.parse.urlencode({'address': address,
                                                         'key': self.api_key})
        dl_session = requests.Session()
        dl_session.mount('https://', HTTPAdapter(max_retries=self.retries))
        address_req = dl_session.get(url)
        address_data = json.loads(address_req.text)

        # Google Geocode only returns the street number if it actually exists
        try:
            if address_data["status"] != "OK":
                print("Google Geocoding Error, status: " + address_data["status"])
                print("Address: " + address)
                return False

            for address_component in address_data['results'][0]['address_components']:
                if ('street_number' in address_component['types']
                        and address_component['short_name'] == street_number_used):
                    return True
        except IndexError:
            print("IndexError for JSON processing, address " + address)
            print(address_req.text)
            raise IndexError("JSON processing failed, address " + address)

        return False

    def download_satellite_image(self, address, **kwargs):
        """

        :param address: A street address or GPS coordinates that will be passed to Google Maps.
        :param kwargs: additional arguments passed to the Google Maps API call
        :return: (str) filename of the saved image (without folder), or empty string on error/duplicate
        """

        image_height = self.image_size + 2*self.crop_size
        image_width = self.image_size
        url = self.build_google_api_url(center=address, size='{}x{}'.format(image_width, image_height), **kwargs)

        # Use session for retries, see stackoverflow post:
        # https://stackoverflow.com/questions/15431044/can-i-set-max-retries-for-requests-request
        dl_session = requests.Session()
        dl_session.mount('https://', HTTPAdapter(max_retries=self.retries))
        image_req = dl_session.get(url)

        if image_req.raw.status != 200:
            msg_for_403 = ''
            if image_req.raw.status == 403 and 'key' not in kwargs:
                msg_for_403 = '\nTo avoid error 403 supply your API key.'
            raise Exception(
                'HTTP Request failed with code {}, reason "{}"'
                .format(image_req.raw.status, image_req.raw.reason)
                + msg_for_403
            )

        expected_content_type = 'image/png'
        if image_req.headers['Content-Type'] != expected_content_type:
            raise Exception("Wrong content type, expecting '{}', received: '{}'"
                            .format(expected_content_type, image_req.headers['Content-Type']))

        now = datetime.datetime.now()
        image_filename = '{:0>4}-{:0>2}-{:0>2}_{:0>2}-{:0>2}-{:0>2} {}.png'\
            .format(now.year, now.month, now.day, now.hour, now.minute, now.second,
                    address_to_filename(address))

        # Check for duplicate
        hash_value = self.get_hash(image_req.content)
        if hash_value in self.hash_sums:
            print('Skipped duplicate of: ' + self.hash_sums[hash_value])
            return ''
        else:
            self.hash_sums[hash_value] = image_filename

        image = Image.open(BytesIO(image_req.content))
        image = image.convert('RGB')
        image_box = image.getbbox()
        image = image.crop((image_box[0],
                            image_box[1]+self.crop_size,
                            image_box[2],
                            image_box[3]-self.crop_size))
        image.save(os.path.join(self.download_folder, image_filename))
        print('Saved: ' + image_filename)

        if self.thumbnail_size > 0:
            image = image.resize((self.thumbnail_size, self.thumbnail_size), resample=PIL.Image.LANCZOS)
            image.save(os.path.join(self.thumbs_folder, self.thumbnail_prefix + image_filename))

        return image_filename


def load_filenames(filenames_csv, skip_headline, skip_index=2):
    first_line_index = 0
    if skip_headline:
        first_line_index = 1
    filenames = [f.split(',') for f in open(filenames_csv, encoding='latin-1').readlines()[first_line_index:] if f.strip()]
    filenames = [(f[0], int(f[1])) for f in filenames if int(f[1]) != skip_index]
    return filenames


def load_data(filenames_csv, folder, image_size, label_map=bool, **kwargs):

    options = {
        'skip_headline': True,
        'horizontal_flip': False,
        'vertical_flip': False,
        'YCbCr': False,
        'featurewise_center': False,
        'featurewise_std_normalization': False,
    }
    options.update(kwargs)

    filenames = load_filenames(filenames_csv, options['skip_headline'], 4)

    sample_count = len(filenames)
    image_versions = 1
    if options['horizontal_flip']:
        sample_count += len(filenames)
        horizontal_flip_index = image_versions
        image_versions += 1
    if options['vertical_flip']:
        sample_count += len(filenames)
        vertical_flip_index = image_versions
        image_versions += 1

    rgb2ycbcr = None
    rgb2ycbcr_shift = None
    if options['YCbCr'] == 'JPEG':
        # Conversion matrix according to JPEG conversion formula
        # https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
        rgb2ycbcr = np.array([
            [0.299000, 0.587000, 0.114000],
            [-0.168736, -0.331264, 0.500000],
            [0.500000, -0.418688, -0.081312]])
        rgb2ycbcr_shift = np.array(
            [0., 128., 128.]
        )
    elif options['YCbCr'] == 'BT601':
        # Conversion matrix according to ITU-R BT.601 standard
        # https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion
        rgb2ycbcr = np.array([
            [65.738, 129.057, 25.064],
            [-37.945, -74.494,  112.439],
            [112.439, -94.154, -18.285]]) / 256.
        rgb2ycbcr_shift = np.array(
            [16., 128., 128.]
        )
    elif options['YCbCr']:
        raise ValueError('Unknown value for option YCbCr: "{}"'.format(str(options['YCbCr'])))

    images_x = np.ndarray((sample_count, image_size, image_size, 3), dtype='float32')
    images_y = np.ndarray((sample_count,), dtype=bool)
    for i, f in enumerate(filenames):
        filename = os.path.join(folder, f[0])
        image = imageio.imread(filename).astype('float32')
        # add :3 in last index for RGBA images
        if image.shape[2] == 4:
            image = image[:, :, :3]

        if options['YCbCr']:
            for x in range(image.shape[0]):
                for y in range(image.shape[1]):
                    image[x, y, :] = rgb2ycbcr.dot(image[x, y, :])+rgb2ycbcr_shift

        images_x[image_versions * i, :, :, :] = image / 255.
        images_y[image_versions * i] = label_map(f[1])

    if options['featurewise_center']:
        for channel in range(3):
            images_x[:, :, :, channel] -= np.mean(images_x[:, :, :, channel])

    if options['featurewise_std_normalization']:
        for channel in range(3):
            images_x[:, :, :, channel] /= np.std(images_x[:, :, :, channel])

    if options['horizontal_flip']:
        for i in range(len(filenames)):
            images_y[image_versions * i + horizontal_flip_index] = images_y[image_versions * i]
            images_x[image_versions * i + horizontal_flip_index, :, :, :] = images_x[image_versions * i, :, ::-1, :]

    if options['vertical_flip']:
        for i in range(len(filenames)):
            images_y[image_versions * i + vertical_flip_index] = images_y[image_versions * i]
            images_x[image_versions * i + vertical_flip_index, :, :, :] = images_x[image_versions * i, ::-1, :, :]

    positive_data = np.sum(images_y)
    print('{} out of {} with solar panel ({:.1f}%)'.format(positive_data, sample_count,
                                                           100.*positive_data/sample_count))
    return images_x, images_y


def fix_filenames(filenames_csv, folder, **kwargs):
    options = {
        'skip_headline': True,
    }
    options.update(kwargs)

    first_line_index = 0
    if options['skip_headline']:
        first_line_index = 1
    filenames = [f.split(',') for f in open(filenames_csv, encoding='latin-1').readlines()[first_line_index:] if f.strip()]
    filenames = [(f[0], int(f[1])) for f in filenames]

    dir_contents = os.listdir(folder)

    renames = 0

    for i, f in enumerate(filenames):
        filename = os.path.join(folder, f[0])
        if not os.path.exists(filename):
            print(re.sub(r'[äöüÄÖÜß]', r'..?', f[0]))
            find_best_match = re.compile(re.sub(r'[äöüÄÖÜß]', r'..?', f[0]))
            for existing_file in dir_contents:
                if find_best_match.search(existing_file):
                    os.rename(os.path.join(folder, existing_file), filename)
                    renames += 1
                    dir_contents.remove(existing_file)
                    break
    return renames


def mode_predict(model, filenames_csv, folder, image_size):
    x_all, y_all = load_data(filenames_csv, folder, image_size,
                      skip_headline=False,
                      featurewise_center=True,
                      featurewise_std_normalization=True)
    y_predict = model.predict(x_all)
    y_predict_labels = np.argmax(y_predict, 1).astype('bool')

    #for i in range(y_predict_labels.shape[0]):
    #    if y_all[i] and (not y_predict_labels[i]):
    #        shutil.copyfile(os.path.join(fs_path, filenames[i][0]), os.path.join('images/false_neg', filenames[i][0]))
    #    if (not y_all[i]) and y_predict_labels[i]:
    #        shutil.copyfile(os.path.join(fs_path, filenames[i][0]), os.path.join('images/false_pos', filenames[i][0]))
    return y_predict_labels
