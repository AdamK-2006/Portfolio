window.leftLayout = "body > div.app > div.app__main > div > div.entity-layout__main > div.entity-layout__left";


//Function for injecting button
function injectButton(target) {
  if (!target || document.querySelector("#generateDocBtn")) return;

  const button = document.createElement("button");
  button.classList.add("ext_button");
  
  button.id = "generateDocBtn";
  button.textContent = "Generate Form";

  //event when button is clicked
  button.addEventListener("click", () => {
    //data is in dict form {"ref": ..., "name": ..., "email": ..., "mobile": ..., "land": ..., "address": ..., "post": ...}
    const data = extractData();
    
    loadTemplateAndGenerateDoc(data);
  });

  if (!target || document.querySelector("#generateDocBtn")) {
    return;
  }
  target.appendChild(button);
}

async function loadTemplates(url) {
  const res = await fetch(url);
  if (!res.ok) {
    console.error('Failed to fetch template list');
    return;
  }
  const files = await res.json();
  return files;
}

//function for injecting dropdown selector
async function injectDropdown(target) {
  if (!target || document.querySelector("#generateDocDrop"))return;

  //create dropdown
  const select = document.createElement("select");
  select.classList.add("ext_dropdown");
  select.id = "generateDocDrop";

  //customize all options
  ///!!!Server Ip goes here
  const BASE_URL = "https://capsuleflask.widefm.co.uk";
  const ext = "/flask/templates";

  const files = await loadTemplates(BASE_URL + ext);
  const business = document.querySelector(window.leftLayout + " > div.panel.contacts-panel");

  let newFiles = [];
  let index = 0;
  for (let file of files) {
    if (file.toLowerCase().includes("business") && business) {
      newFiles.splice(index, 0, file);
      index ++;
    } else if (file.toLowerCase().includes("residential") && !business) {
      newFiles.splice(index, 0, file);
      index ++;
    } else
      newFiles.push(file);
  }

  const option = document.createElement("option");
  option.value = "Choose Template";
  option.textContent = "Choose Template";
  select.appendChild(option);
  
  newFiles.forEach(file => {
    const option = document.createElement('option');
    option.value = file;
    option.textContent = file;
    select.appendChild(option);
  });

  if (!target || document.querySelector("#generateDocDrop")) {
    return;
  }
  target.appendChild(select);
}

//function that extracts information from webpage
function extractData() {
  let name = "-";
  let company = "-"
  let mobile = "-";
  let land = "-";
  let address = "-";
  let post = "-";
  let email = "-";
  let ref = "-";
  try {
    //extracts name
    const namePath = "div:nth-child(1) > div > div.party-details.primary-party-details > div.party-details__text > span";
    const nameEl = document.querySelector(window.leftLayout + " > " + namePath);
    if (!document.querySelector(window.leftLayout + "div.panel.contacts-panel")){
      const nameArr = nameEl?.textContent?.trim().split(/\r?\n/);
      if (nameArr.length > 1)
        name = nameArr[1].trim();
      else
        name = nameArr[0].trim();
    } else {
      company = nameEl?.textContent?.trim();
      const contactPanel = "div.panel.contacts-panel > div.panel__content > ul";
      const contact = document.querySelector(window.leftLayout + " > " + contactPanel);
      if (contact) {
        const contactNameEl = document.querySelector(contactPanel + " > li:nth-child(1) > div.contacts-panel__item-container > a")
        const nameArr = contactNameEl?.textContent?.trim().split(/\r?\n/);
        if (nameArr.length > 1)
          name = nameArr[1].trim();
        else
          name = nameArr[0].trim();
      }
    }

    if (!name)
      name = "-";
    if (!company)
      company = "-";
    //list of contact info, includes phone, email and address in that order
    const ul = document.querySelector(window.leftLayout + " > div:nth-child(1) > div > ul");
    const ulLen = ul ? ul.querySelectorAll("li").length : 0;
    if (!ul) {
      alert("No contact details found on page");
      throw Error;
    }
    //the following retrieves phone numbers, email and address
    //as phone numbers are first, mobile and land are retrieved sequentially until an invalid number is found
    //email is second to last and address is last so they can be retrieved via the last index
    //the following contains checks if one or both of email and address are missing
    const detailsList = "div:nth-child(1) > div > ul";
    detailsItemPath = "div > div.copy__contents.contact-detail__detail";
    for (let i = 1; i <= ulLen; i++) {
      let currentElem = document.querySelector(window.leftLayout + " > " + detailsList + " > li:nth-child("+i+") > " + detailsItemPath);
      let current = currentElem?.textContent?.trim()
      if (current.substring(0, 1) == "0") {
        if (current.substring(0, 2) == "07" && mobile == "-" && !current.includes("@") && !current.includes("."))
          mobile = current;
        else if (/^0[0-68-9]/.test(current) && land == "-" && !current.includes("@") && !current.includes("."))
          land = current;
      } else  if (current.includes("@")) {
        email = current;
        break;
      }
    }

    const addressElem = document.querySelector(window.leftLayout + " > " + detailsList + " > li:nth-child("+ulLen+") > " + detailsItemPath);
    address = addressElem?.textContent?.trim();
    if (!address)
      address = "-";
    else if (address.substring(0, 1) == "0" || address == land || address == mobile) //phone
      address = "-";
    else if (address.includes(".") || address.includes("@")) //website and email
      address = "-"
    else if (/[a-zA-Z]/.test(address)) {
      //if address is not missing
      //finds postcode in address
      const match = address.toUpperCase().match(/([A-Z][A-HJ-Y]?\d[A-Z\d]? ?\d[A-Z]{2}|GIR ?0A{2})/);
      post = match? match[0] : "-";

      address = address.replace(", " + post, '');
      address = address.replace(", " + post.toLowerCase(), '');
      address = address.replace(post, '');
      address = address.replace(post.toLowerCase(), '');
      if (address == "")
        address = "-";
    }

    const dlElements = document.querySelectorAll('dl.custom-values');
    for (let i = dlElements.length - 1; i >= 0; i--) {
      const dl = dlElements[i];
      const valueSpan = dl.querySelector('span.custom-value');
      const keyElem = dl.querySelector('dt.key-value__term');
      
      if (valueSpan && keyElem) {
        if (keyElem.textContent.trim() == "Order Number" && valueSpan.textContent.trim().length == 6) {
          ref = valueSpan.textContent.trim();
          break;
        }
      }
    }
  } catch (err) {
    console.error("Error extracting data from page:", err);
    alert("Failed to extract contact data from the page");
  }
  const data = {"ref": ref, "name": name, "company": company, 
    "address": address, "post": post, "mobile": mobile, "land" : land, "email": email};
  return data;
}

function contains(arr, item) {
    for (let n of arr) {
        if (n == item) {
            return true;
        }
    }
    return false;
}

function extractTagsFromXml(xmlText) {
  // Matches {tag}, ${tag}, or {{tag}} depending on your template syntax
  const tagRegex = /\{[\$]?([\w\d_.-]+)\}/g;
  const tags = new Set();
  let match;

  while ((match = tagRegex.exec(xmlText)) !== null) {
    tags.add(match[1]); // match[1] is the tag inside the braces
  }

  return Array.from(tags);
}

//function to load a template and generate a doc from it
async function loadTemplateAndGenerateDoc(data) {
  //retieves file name from dropdown
  const select = document.getElementById("generateDocDrop");
  if (!select || !select.selectedIndex || select.selectedIndex == 0) {
    alert("No template selected or dropdown not found.");
    return;
  }
  const filename = select.options[select.selectedIndex].text;

  //loads template file
  let arrayBuffer;
  try {
    ///!!!Server Ip goes here
    const serverUrl = "https://capsuleflask.widefm.co.uk";
    const ext = "/flask/templates"
    const response = await fetch(serverUrl + ext + "/" + filename);
    if (!response.ok) 
      throw new Error("Template fetch failed");
    arrayBuffer = await response.arrayBuffer();
  } catch (err) {
    console.error("Failed to load template file");
    alert("Failed to load the selected template file");
    return;
  }
  
  //set data

  if (!data.name || data.name == "-") {
    data.name = "-";
  }
  if (!data.address|| data.address == "-") {
    data.address = "-"
  }
  else if (!data.post || data.post == "-") {
    data.post = "-"
  }

  const zip = new PizZip(arrayBuffer);
  const documentXml = zip.file("word/document.xml").asText();
  if (!contains(extractTagsFromXml(documentXml), "land") && data.mobile == "-")
    data.mobile = data.land;

  const doc = new window.docxtemplater().loadZip(zip);
  doc.setData(data);

  try {
    doc.render();
  } catch (error) {
    //output error if there are any issues
    if (error.properties && error.properties.errors) {
      error.properties.errors.forEach(err => {
        console.error(`Docx template error: ${err.message}`, err);
      });
    } else {
      console.error("Docx template error:", error);
    }
    return;
  }

  //downloads file
  const out = doc.getZip().generate({ type: "blob" });
  let name;
  if (data.ref == "-")
    name = filename.replace(".docx", '');
  else
    name = (filename.replace(".docx", '') + " " + data.ref);
  saveAs(out, `${name || "Contract"}.docx`);
}

//function that waits for selector (page) to be loaded before doing callback (injecting elements)
function waitForElement(selector, callback) {
  const observer = new MutationObserver(() => {
    const el = document.querySelector(selector);
    if (el) {
      observer.disconnect();
      callback(el);
    }
  })

  observer.observe(document.body, {childList: true, subtree: true});
}

//injects css if needed
function injectCSSIfNeeded() {
  const styleId = 'capsuledoc-styles';
  if (!document.getElementById(styleId)) {
  const style1 = document.createElement("link");
  style1.rel = "stylesheet";
  style1.href = chrome.runtime.getURL("styles/button.css");
  style1.id = styleId;

  const style2 = document.createElement("link");
  style2.rel = "stylesheet";
  style2.href = chrome.runtime.getURL("styles/dropdown.css");

  document.head.appendChild(style1);
  document.head.appendChild(style2);
  }
}

injectCSSIfNeeded();

// Start observing for the target container before injecting button
waitForElement("body > div.app > div.app__main > div > div.entity-layout__main > div.entity-layout__left", async (target) => {
  injectButton(target);
  await injectDropdown(target);
});