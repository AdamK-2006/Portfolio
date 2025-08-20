//checks if tab has loaded, if it has content.js is run
//this was added as the button and dropdown would not be injected unless the page was refreshed, not when it was initially opened

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.url.match(/https:\/\/.*\.capsulecrm\.com\/party\//)) {
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      files: ["utils/pizzip.min.js", "utils/docxtemplater.min.js", "utils/FileSaver.min.js", "content.js"]
    });
  }
});