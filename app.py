import streamlit as st
from PIL import Image
import numpy as np
import pickle
from pathlib import Path
from bs4 import BeautifulSoup as bs
import requests
from urllib.parse import quote
import base64
from io import BytesIO


# --- GENERAL SETTINGS ---
PAGE_TITLE = "Laptop Price Predictor"
PAGE_ICON = "💻"
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)


def add_bg_from_local(image_files):
    with open(image_files[0], "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    with open(image_files[1], "rb") as image_file:
        encoded_string1 = base64.b64encode(image_file.read())
    with open(image_files[2], "rb") as image_file:
        encoded_string2 = base64.b64encode(image_file.read())
    st.markdown(
        """
    <style>
      .stApp {
          background-image: url(data:image/png;base64,"""+encoded_string.decode()+""");
          background-size: cover;
      }
      .css-6qob1r.e1fqkh3o3 {
        background-image: url(data:image/png;base64,"""+encoded_string1.decode()+""");
        background-size: cover;
        background-repeat: no-repeat;
      }
      .css-1avcm0n.e8zbici2 {
        background-image: url(data:image/png;base64,"""+encoded_string2.decode()+""");
        background-size: cover;
        background-repeat: no-repeat;
      }
    </style>""",
        unsafe_allow_html=True
    )


add_bg_from_local(
    [r'10340256_13077.jpg', r'10340256_13077.jpg', r'10340256_13077.jpg'])


# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"

# --- LOAD CSS---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

# Heading of the Page
st.markdown("<h1 style='text-align: center; text-decoration: none;'>Laptop Predictor</h1>",
            unsafe_allow_html=True)

# All inputs:
# brand
brand_options = df['Company'].unique().tolist()
brand_options.insert(0, 'Select an option')
company = st.selectbox('Brand', brand_options)

# type of laptop
type_options = df['TypeName'].unique().tolist()
type_options.insert(0, 'Select an option')
laptop_type = st.selectbox('Type', type_options)

# Ram
ram_options = [2, 4, 6, 8, 12, 16, 24, 32, 64]
ram_options.insert(0, 'Select an option')
ram = st.selectbox('RAM(in GB)', ram_options)

# weight
weight = st.number_input('Weight of the Laptop(kg)')

# Touchscreen
touchscreen_options = ['No', 'Yes']
touchscreen_options.insert(0, 'Select an option')
touchscreen = st.selectbox('Touchscreen', touchscreen_options)

# IPS
ips_options = ['No', 'Yes']
ips_options.insert(0, 'Select an option')
ips = st.selectbox('IPS', ips_options)

# screen size
screen_size = st.number_input('Screen Size (in inch)')

# resolution
resolution = st.selectbox('Screen Resolution', [
                          '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# cpu
cpu_options = df['Cpu brand'].unique().tolist()
cpu_options.insert(0, 'Select an option')
cpu = st.selectbox('CPU', cpu_options)

# HDD
hdd_options = [0, 128, 256, 512, 1024, 2048, 4096]
hdd_options.insert(0, 'Select an option')
hdd = st.selectbox('HDD(in GB)', hdd_options)

# SSD
ssd_options = [0, 128, 256, 512, 1024, 2048]
ssd_options.insert(0, 'Select an option')
ssd = st.selectbox('SSD(in GB)', ssd_options)

# GPU
gpu_options = df['Gpu brand'].unique().tolist()
gpu_options.insert(0, 'Select an option')
gpu = st.selectbox('GPU', gpu_options)

# os
if company == 'Apple':
    os_options = ['Mac']
else:
    os_options = [option for option in df['os'].unique() if option != 'Mac']
os = st.selectbox('OS', ['Select an OS'] + os_options)

st.text("")

# Add a expander for DISCLAIMER
expand_sidebar = st.checkbox("DISCLAIMER")
if expand_sidebar:
    st.markdown(
        "<style> .caption-text { color: white; } </style>", unsafe_allow_html=True)
    st.caption("<p class='caption-text'>The laptop price predictions provided by this application are generated using historical data and machine learning algorithms. It's important to note that actual prices may vary due to market dynamics and other factors. We encourage users to consider this information as a helpful reference and complement it with additional research and expert advice. Our app aims to assist users in making informed purchasing decisions by providing valuable insights into laptop prices and provide recommendations.</p>", unsafe_allow_html=True)

st.text("")
col1, col2, col3 = st.columns(3)
with col1:
    predict_price_checked = st.button('Predict Price')
with col3:
    recommendation_checked = st.button('Recommendations')

# Prediction
if predict_price_checked:
    # query
    if company == 'Select an option' or laptop_type == 'Select an option' or ram == 'Select an option' or \
            weight < 0.5 or touchscreen == 'Select an option' or ips == 'Select an option' or \
            screen_size == 0 or resolution == 'Select a resolution' or cpu == 'Select a CPU' or \
            (hdd == 'Select an option' and ssd == 'Select an option') or (hdd == 0 and ssd == 0) or gpu == 'Select a GPU' or os == 'Select an OS':
        st.error('Please select all specifications.')
    else:
        ppi = None
        if touchscreen == 'Yes':
            touchscreen = 1
        else:
            touchscreen = 0

        if ips == 'Yes':
            ips = 1
        else:
            ips = 0

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        try:
            ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
            query = np.array([company, laptop_type, ram, weight, touchscreen,
                             ips, ppi, cpu, hdd, ssd, gpu, os], dtype='object')

            query = query.reshape(1, 12)
            st.title("The predicted price of this configuration is " +
                     str(int(np.exp(pipe.predict(query)[0]))))
        except Exception as e:
            if hdd == 'Select an option':
                st.write("Please select 0 if your laptop doesn't contain any HDD")
            elif ssd == 'Select an option':
                st.write("Please select 0 if your laptop doesn't contain any SSD")
            else:
                st.write("Please don't enter 0, mention some value!")

if recommendation_checked:
    if company == 'Select an option' or laptop_type == 'Select an option' or ram == 'Select an option' or \
            weight < 0.5 or touchscreen == 'Select an option' or ips == 'Select an option' or \
            screen_size == 0 or resolution == 'Select a resolution' or cpu == 'Select a CPU' or \
            (hdd == 'Select an option' and ssd == 'Select an option') or (hdd == 0 and ssd == 0) or \
            gpu == 'Select a GPU' or os == 'Select an OS':
        st.error('Please select all specifications.')
    else:
        # Adding %20 between spaces in option
        encoded_lt = quote(laptop_type)
        encoded_cpu = quote(cpu)
        # To set the search on url based on the specification so users gets the perfect match
        if touchscreen.lower() == 'yes':
            if hdd != 'Select an option' and hdd != 0:
                url = "https://www.flipkart.com/search?q=" + str(company) + "%20" + str(encoded_lt) + "%20" + str(
                    ram) + "gb%20" + "touchscreen" + "%20" + str(encoded_cpu) + "%20" + str(hdd) + \
                    "gb%20hdd%20" + str(os) + "%20" + str(
                    gpu) + "&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off"

            else:
                url = "https://www.flipkart.com/search?q=" + str(company) + "%20" + str(encoded_lt) + "%20" + str(
                    ram) + "gb%20ram%20" + "touchscreen" + "%20" + str(encoded_cpu) + "%20" + str(ssd) + \
                    "gb%20ssd%20" + str(os) + "%20" + str(
                    gpu) + "&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off"

        elif touchscreen.lower() == 'no':
            if hdd != 'Select an option' and hdd != 0:
                url = "https://www.flipkart.com/search?q=" + str(company) + "%20" + str(encoded_lt) + "%20" + str(
                    ram) + "gb%20ram%20" + str(encoded_cpu) + "%20" + str(hdd) + \
                    "gb%20hdd%20" + str(
                    os) + "&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off"

            else:
                url = "https://www.flipkart.com/search?q=" + str(company) + "%20" + str(encoded_lt) + "%20" + str(
                    ram) + "gb%20ram%20" + str(encoded_cpu) + "%20" + str(ssd) + \
                    "gb%20ssd%20" + str(
                    os) + "&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off"

        else:  # No needed for this else as it won't move forward if user doesn't even select one specification properly
            st.error("Your Laptop is Touch Screen or not? Please select an option")

        try:
            data = requests.get(url)
            plain_text = data.text
            soup = bs(plain_text, "html.parser")

            # For Title (Top 3 Recommendations)
            col1, col2, col3 = st.columns(3)
            with col2:
                st.markdown(
                    "<p align='center'><strong style='font-size: 24px;'>Top 3 Recommendations</strong></p>", unsafe_allow_html=True)

            # Create a placeholder element to display the result
            result_placeholder = st.empty()
            # show spinner while the webscraping code is running
            with st.spinner("Searching..."):
                # To Scrape Image, Name, Price and link for the first three options
                for i, product in enumerate(soup.find_all("div", class_="_2kHMtA", limit=3)):
                    # Extract the image URL
                    img_tag = product.find("img")
                    if img_tag and 'src' in img_tag.attrs:
                        image_url = img_tag['src']
                        response = requests.get(image_url)
                        image_data = BytesIO(response.content)
                        image = Image.open(image_data)
                        resized_image = image.resize((215, 130))

                        # Extract the name, price and link
                        name = product.find("div", class_="_4rR01T").text
                        price = product.find(
                            "div", class_="_30jeq3 _1_WHN1").text
                        a_tag = product.find("a", class_="_1fQZEK")
                        href_value = a_tag["href"]

                        # Create columns for each option
                        col1, col2, col3 = st.columns(3)

                        def image_to_base64(image):
                            buffered = BytesIO()
                            image.save(buffered, format="PNG")
                            return base64.b64encode(buffered.getvalue()).decode("utf-8")

                        # As link was opening in local host so to avoid that
                        real_link = "https://www.flipkart.com" + href_value

                        # Display the image, name, and price in respective columns
                        with col1:
                            image_html = f"<a href='{real_link}' target='_blank'><img src='data:image/png;base64,{image_to_base64(resized_image)}'></a>"
                            st.markdown(image_html, unsafe_allow_html=True)

                        with col2:
                            st.markdown(
                                f"<p align='center'><strong><a href='{real_link}' target='_blank' style='text-decoration: none; color: inherit;'>{name}</a></strong></p>",
                                unsafe_allow_html=True,
                            )

                        with col3:
                            st.markdown(
                                f"<p align='center'><strong>{price}</strong></p>", unsafe_allow_html=True)

                    else:
                        st.error(
                            "Some required elements not found in the product.")

            result_placeholder.write()  # Update the placeholder element with the result

        except Exception as e:
            st.error(f"An error occurred during scraping, Please ensure that the specifications are accurately selected. Avoid inputting any arbitrary values that may affect the results.")
