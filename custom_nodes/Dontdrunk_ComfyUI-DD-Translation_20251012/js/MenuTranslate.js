import { 
  containsChineseCharacters, 
  nativeTranslatedSettings,
  error
} from "./utils.js";

/**
 * 翻译执行器类
 */
class TExe {
  static T = null;

  /**
   * 翻译指定文本
   * @param {string} txt 需要翻译的文本
   * @returns {string|null} 翻译后的文本或null
   */
  MT(txt) {
    // 如果文本已经包含中文字符，跳过翻译
    if (containsChineseCharacters(txt)) {
      return null;
    }
    return this.T?.Menu?.[txt] || this.T?.Menu?.[txt?.trim?.()];
  }

  constructor() {
    // 不需要翻译的CSS类列表
    this.excludeClass = ["lite-search-item-type"];
    // 记录已注册的观察者，便于后续管理
    this.observers = [];
  }

  /**
   * 检查是否需要跳过翻译
   * @param {HTMLElement} node DOM节点
   * @returns {boolean} 是否需要跳过
   */
  tSkip(node) {
    try {
      // 判断node.classList 是否包含 excludeClass中的一个
      return this.excludeClass.some((cls) => node.classList?.contains(cls));
    } catch (e) {
      // 如果出错，默认不跳过
      return false;
    }
  }
  translateKjPopDesc(node) {
    try {
      let T = this.T;
      if (!T) return false;
      if (!node || !node.querySelectorAll) return false;
      if (!node?.classList?.contains("kj-documentation-popup")) return false;
      
      const allElements = node.querySelectorAll("*");
      for (const ele of allElements) {
        this.replaceText(ele);
      }
      
      return true;
    } catch (e) {
      error("翻译KJ弹窗出错:", e);
      return false;
    }
  }
  translateAllText(node) {
    try {
      let T = this.T;
      if (!T) return;
      if (!node || !node.querySelectorAll) return;
      
      const allElements = node.querySelectorAll("*");
      for (const ele of allElements) {
        if (ele.textContent && nativeTranslatedSettings.includes(ele.textContent)) {
          continue;
        }
        this.replaceText(ele);
      }
    } catch (e) {
      error("翻译所有文本出错:", e);
    }
  }

  /**
   * 替换文本内容为翻译后的文本
   * @param {Node} target 目标节点
   */
  replaceText(target) {
    try {
      if (!target) return;
      if (!this.T) return;
      if (this.tSkip(target)) return;
      
      // 如果节点的内容是原生已翻译的设置项，跳过翻译
      if (target.textContent && nativeTranslatedSettings.includes(target.textContent)) {
        return;
      }
      
      // 处理子节点
      if (target.childNodes && target.childNodes.length) {
        // 创建一个副本来遍历，避免在遍历过程中修改导致问题
        const childNodes = Array.from(target.childNodes);
        for (const childNode of childNodes) {
          this.replaceText(childNode);
        }
      }
      
      // 处理当前节点
      if (target.nodeType === Node.TEXT_NODE) {
        // 文本节点
        if (target.nodeValue && !containsChineseCharacters(target.nodeValue)) {
          const translated = this.MT(target.nodeValue);
          if (translated) {
            target.nodeValue = translated;
          }
        }
      } else if (target.nodeType === Node.ELEMENT_NODE) {
        // 元素节点
        
        // 处理 title 属性
        if (target.title && !containsChineseCharacters(target.title)) {
          const titleTranslated = this.MT(target.title);
          if (titleTranslated) {
            target.title = titleTranslated;
          }
        }

        // 处理按钮值
        if (target.nodeName === "INPUT" && target.type === "button" && 
            !containsChineseCharacters(target.value)) {
          const valueTranslated = this.MT(target.value);
          if (valueTranslated) {
            target.value = valueTranslated;
          }
        }

        // 处理文本内容
        if (target.innerText && !containsChineseCharacters(target.innerText)) {
          const innerTextTranslated = this.MT(target.innerText);
          if (innerTextTranslated) {
            target.innerText = innerTextTranslated;
          }
        }
        
        // 处理select和option元素
        if (target.nodeName === "SELECT") {
          // 确保翻译下拉框中的选项
          Array.from(target.options).forEach(option => {
            if (option.text && !containsChineseCharacters(option.text)) {
              const optionTextTranslated = this.MT(option.text);
              if (optionTextTranslated) {
                option.text = optionTextTranslated;
              }
            }
          });
        }
      }
    } catch (e) {
      error("替换文本出错:", e);
    }
  }
    cleanupObservers() {
    try {
      this.observers.forEach(observer => {
        if (observer && typeof observer.disconnect === 'function') {
          observer.disconnect();
        }
      });
      this.observers = [];
    } catch (e) {
      error("清理观察者出错:", e);
    }
  }
}

// 创建翻译执行器实例
let texe = new TExe();

export function applyMenuTranslation(T) {
  try {
    texe.cleanupObservers();
    texe.T = T;
    
    texe.translateAllText(document.querySelector(".litegraph"));
    
    const bodyObserver = observeFactory(document.querySelector("body.litegraph"), (mutationsList) => {
      for (let mutation of mutationsList) {
        for (const node of mutation.addedNodes) {
          if (node.classList?.contains("comfy-modal")) {
            texe.translateAllText(node);
            observeModalNode(node);
          } else if (node.classList?.contains("p-dialog-mask")) {
            const dialog = node.querySelector(".p-dialog");
            if (dialog) {
              texe.translateAllText(dialog);
              observeFactory(dialog, handleSettingsDialog, dialog?.role === "dialog");
            }
          } else {
            texe.translateAllText(node);
          }
        }
      }
    }, true);
    
    texe.observers.push(bodyObserver);
    
    document.querySelectorAll(".comfy-modal").forEach(node => {
      observeModalNode(node);
    });
    
    if (document.querySelector(".comfyui-menu")) {
      const menuObserver = observeFactory(document.querySelector(".comfyui-menu"), handleComfyNewUIMenu, true);
      texe.observers.push(menuObserver);
    }
    
    document.querySelectorAll(".comfyui-popup").forEach(node => {
      const popupObserver = observeFactory(node, handleComfyNewUIMenu, true);
      texe.observers.push(popupObserver);
    });
    
    handleHistoryAndQueueButtons();
    handleSettingsDialog();
    setupSearchBoxObserver();
  } catch (e) {
    error("应用菜单翻译出错:", e);
  }
}

/**
 * 观察者工厂函数
 * @param {HTMLElement} observeTarget 观察目标
 * @param {Function} fn 回调函数
 * @param {boolean} subtree 是否观察子树
 * @returns {MutationObserver} 观察者实例
 */
export function observeFactory(observeTarget, fn, subtree = false) {
  if (!observeTarget) return null;
  try {
    const observer = new MutationObserver(function (mutationsList, observer) {
      fn(mutationsList, observer);
    });

    observer.observe(observeTarget, {
      childList: true,
      attributes: true,
      subtree: subtree,
    });
    return observer;
  } catch (e) {
    error("创建观察者出错:", e);
    return null;
  }
}

/**
 * 处理模态框节点
 * @param {HTMLElement} node 模态框节点
 */
function observeModalNode(node) {
  const observer = observeFactory(node, (mutationsList) => {
    for (let mutation of mutationsList) {
      texe.translateAllText(mutation.target);
    }
  });
  if (observer) {
    texe.observers.push(observer);
  }
}

/**
 * 处理ComfyUI新版UI菜单
 * @param {MutationRecord[]} mutationsList 变更记录列表
 */
function handleComfyNewUIMenu(mutationsList) {
  for (let mutation of mutationsList) {
    texe.translateAllText(mutation.target);
  }
}

function handleHistoryAndQueueButtons() {
  const viewHistoryButton = document.getElementById("comfy-view-history-button");
  const viewQueueButton = document.getElementById("comfy-view-queue-button");

  [viewHistoryButton, viewQueueButton].filter(Boolean).forEach(btn => {
    const observer = observeFactory(btn, (mutationsList) => {
      for (let mutation of mutationsList) {
        if (mutation.type === "childList") {
          const translatedValue = texe.MT(mutation.target.textContent);
          if (translatedValue) {
            mutation.target.innerText = translatedValue;
          }
        }
      }
    });
    if (observer) {
      texe.observers.push(observer);
    }
  });
  
  if (document.querySelector(".comfy-menu")) {
    const menuObserver = observeFactory(document.querySelector(".comfy-menu"), handleViewQueueComfyListObserver);
    if (menuObserver) {
      texe.observers.push(menuObserver);
    }

    const comfyLists = document.querySelector(".comfy-menu").querySelectorAll(".comfy-list");
    if (comfyLists.length > 0) {
      const list0Observer = observeFactory(comfyLists[0], handleViewQueueComfyListObserver);
      if (list0Observer) {
        texe.observers.push(list0Observer);
      }
      
      if (comfyLists.length > 1) {
        const list1Observer = observeFactory(comfyLists[1], handleViewQueueComfyListObserver);
        if (list1Observer) {
          texe.observers.push(list1Observer);
        }
      }
    }
  }
}

/**
 * 处理视图队列和Comfy列表观察者
 * @param {MutationRecord[]} mutationsList 变更记录列表
 */
function handleViewQueueComfyListObserver(mutationsList) {
  for (let mutation of mutationsList) {
    texe.replaceText(mutation.target);
    if (mutation.type === "childList" && mutation.addedNodes.length > 0) {
      for (const node of mutation.addedNodes) {
        texe.replaceText(node);
      }
    }
  }
}

/**
 * 处理设置对话框
 */
function handleSettingsDialog() {
  const comfySettingDialog = document.querySelector("#comfy-settings-dialog");
  if (!comfySettingDialog) return;

  // 老版设置面板的翻译
  if (comfySettingDialog?.querySelector("tbody")) {
    const observer = observeFactory(comfySettingDialog.querySelector("tbody"), (mutationsList) => {
      for (let mutation of mutationsList) {
        if (mutation.type === "childList" && mutation.addedNodes.length > 0) {
          translateSettingDialog(comfySettingDialog);
        }
      }
    });
    if (observer) {
      texe.observers.push(observer);
    }
  }

  // 新版设置面板处理
  const newSettingsPanels = document.querySelectorAll(".p-dialog-content, .p-tabview-panels");
  for (const panel of newSettingsPanels) {
    const observer = observeFactory(panel, handleNewSettingsObserver, true);
    if (observer) {
      texe.observers.push(observer);
    }
  }
  
  // 初始翻译
  translateSettingDialog(comfySettingDialog);
}

/**
 * 处理新版设置观察者
 * @param {MutationRecord[]} mutationsList 变更记录列表
 */
function handleNewSettingsObserver(mutationsList) {
  for (let mutation of mutationsList) {
    if (mutation.type === "childList") {
      for (const node of mutation.addedNodes) {
        texe.translateAllText(node);
      }
    }
  }
}

/**
 * 翻译设置对话框
 * @param {HTMLElement} comfySettingDialog 设置对话框
 */
function translateSettingDialog(comfySettingDialog) {
  if (!comfySettingDialog) return;
  
  const comfySettingDialogAllElements = comfySettingDialog.querySelectorAll("*");
  for (const ele of comfySettingDialogAllElements) {
    // 跳过已经有中文的元素
    if (containsChineseCharacters(ele.innerText) || 
        nativeTranslatedSettings.includes(ele.innerText)) {
      continue;
    }
    
    let targetLangText = texe.MT(ele.innerText);
    let titleText = texe.MT(ele.title);
    if (titleText) ele.title = titleText;
    if (!targetLangText) {
      if (ele.nodeName === "INPUT" && ele.type === "button") {
        targetLangText = texe.MT(ele.value);
        if (!targetLangText) continue;
        ele.value = targetLangText;
      }
      continue;
    }
    texe.replaceText(ele);
  }
}

/**
 * 设置搜索框观察者
 */
function setupSearchBoxObserver() {
  const searchObserver = observeFactory(document.querySelector(".litegraph"), (mutationsList, observer) => {
    // 存储搜索框观察者的引用
    if (!observer.searchBoxObservers) {
      observer.searchBoxObservers = [];
    }
    
    for (let mutation of mutationsList) {
      // 清理旧的搜索框观察者
      if (mutation.removedNodes.length > 0 && observer.searchBoxObservers.length > 0) {
        observer.searchBoxObservers.forEach(ob => {
          if (ob && typeof ob.disconnect === 'function') {
            ob.disconnect();
          }
        });
        observer.searchBoxObservers = [];
        continue;
      }
      
      // 处理新添加的搜索框
      for (const sb of mutation.addedNodes) {
        if (!sb || !sb.querySelector) continue;
        const helper = sb.querySelector(".helper");
        if (!helper) continue;
        
        // 观察搜索助手内容变化
        const helperObserver = observeFactory(helper, (mutationsList) => {
          for (let mutation of mutationsList) {
            for (const item of mutation.addedNodes) {
              if (item.innerText && texe.T.Nodes[item.innerText]) {
                item.innerText = texe.T.Nodes[item.innerText]["title"] || item.innerText;
              }
            }
          }
        });
        
        if (helperObserver) {
          observer.searchBoxObservers.push(helperObserver);
        }
        
        // 翻译现有搜索项
        for (let item of helper.querySelectorAll(".lite-search-item")) {
          if (item.innerText && texe.T.Nodes[item.innerText]) {
            item.innerText = texe.T.Nodes[item.innerText]["title"] || item.innerText;
          }
        }
      }
    }
  });
  
  if (searchObserver) {
    texe.observers.push(searchObserver);
  }
}
