# Capsule Document Generator
Note: This extension was built specifically for internal use at WideFM and is not intended for general use.\
\
This project is a Google extension that extracts customer information from Capsule (A customer information base) and generates a contract document from a list of templates specified by the user. 

### The Problem
This is a project I worked on for one of my previous employers. The issue presented to me was when any of the sales team wished to create a contract for a customer on capsule, they would manually have to find the correct template and fill in the details such as name, address etc.\
\
With this extension, users can instead select which template they would like to generate, and then with the click of a button the extension produces a Word document.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [License](#license)

## Installation
1. Download the `capsule-zip.zip` file.
2. Extract the folder and take note of its location.
3. Open **Google Chrome** and go to `chrome://extensions/`.
4. Enable **Developer mode** (at the top right).
5. Click **Load unpacked** (at the top left) and select the extracted project folder.
6. The extension should now be working!

## Usage
1) Go to the **WideFM** capsule page.
2) Navigate to a person/organisation `https://widefm.capsulecrm.com/party/*`.
3) The button and dropdown can be found on the left hand side under the customer information.
##### View of Contact page:
<img width="1904" height="829" alt="image" src="https://github.com/user-attachments/assets/6efb3e1d-b343-477e-85b6-0c4f9bbbf412" />

## Features
#### Extension
- Button that looks identical to other buttons found on the page, seamless design.
- Dropdown for selecting templates created by WideFM.
#### Template & Backend
- Templates found on an internal server.
- Meaningfully named tags for customer information inserted on each template, e.g. {name}, {address}, {post}, {mobile} etc.
- A Python Flask server that consistently runs on the server. There are two functions as part of this:
  * One which returns an individual template.
  * Another that returns a list of template names, this is for the dropdown menu.

## License
This project is proprietary and was developed for internal use at WideFM. A license will be specified if released for public use.

