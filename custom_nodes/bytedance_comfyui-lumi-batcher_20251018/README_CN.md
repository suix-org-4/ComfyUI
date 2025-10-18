简体中文 | [English](./README.md)

<div align="center">

# 🚀 ComfyUI Lumi Batcher

**By 字节跳动智能创作团队**

<h4 align="center">
<div align="center">
<img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
<img src="https://img.shields.io/badge/License-GPL 3.0-green.svg" alt="License">
<img src="https://img.shields.io/github/stars/bytedance/comfyui-lumi-batcher?color=yellow" alt="Stars">
<img src="https://img.shields.io/github/issues/bytedance/comfyui-lumi-batcher?color=orange" alt="Issues">
<img src="https://img.shields.io/badge/python-3.10%2B-red.svg" alt="Python">
</h4>

</div>

https://github.com/user-attachments/assets/9d4588b5-696f-4b5c-b01b-4b6def5056cf

&nbsp;

---

&nbsp;

## 📌 概括

**ComfyUI-Lumi-Batcher 是专为 ComfyUI 设计的批量处理扩展插件，旨在提升 AIGC 创作效率。**

&nbsp;

## 😭 你是否在经历这些创作难题

❌ _模型选择困难症晚期_
反复替换模型手动跑图，3 小时试不出最佳风格

❌*参数调试逼疯设计师*
手动调整尺寸/权重/采样步数，1 张图改 20 版甲方仍不满意

❌ _素材管理如大海捞针_
生成 100 张图混乱命名，跑图 10 分钟找图 2 小时

&nbsp;

---

&nbsp;

## ☀️Comfyui-Lumi-Batcher 如何帮你解决难题

ComfyUI-Lumi-Batcher 之于 comfyui，用途范围超过 xyz pilot 之于 webui，是一个调参抽卡、批量出素材的利器。

- **不止“xyz”简单维度，全部参数随心配**：不局限在 xyz 三维参数的交叉调试，而是 workflow 内的所有参数均可交叉，一次完成所有调试；
- **多组参数自由组合，场景个性配置**： 批量调试可以是多个参数的组合体，如「商品图 + 对应 prompt」 X 不同基模；
- **多维表格，自由抽卡预览或批量导出**：不同参数的聚合浏览方式，可视化工具让你对工作流效果运筹帷幄。

&nbsp;

---

&nbsp;

## 😁 为何选择 Comfyui-Lumi-Batcher

- 🔥**易于使用**：丝滑交互降低学习成本，能跑 Comfyui 就能上手工具
- 🔥**高效创作**：「单次设置参数」代替「反复输入参数」、「智能化结果管理」代替「手动下载效果对比」，创作效率跃迁
- 🔥**多模态支持**：深入各类创作场景，无论你在进行文本、图像还是视频创作，都能助你一臂之力

&nbsp;

---

&nbsp;

## ⭐️ 版本更新

- ✂️ **Version 1.0.7**：支持 ComfyUI 官方最新版改动，保证批量工具使用
- ☀️ **Version 1.0.6**：兼容官方 0.3.44 版本针对 validate_prompt 的改动
- ☕️ **Version 1.0.5**：支持使用 api 节点场景下的批量，支持自定义 output 目录，批量支持 was-node-suite-comfyui 插件的 Image Save 输出节点
- 👁 **Version 1.0.4**：优化工作流发起批量任务时的检测逻辑不再实际发起一次任务推理,支持使用 cg-use-everywhere 插件场景的批量
- 🔍 **Version 1.0.3**：任务管理支持删除功能，自定义参数支持魔法表达式快速输入，单图结果预览支持下载功能
- 📚 **Version 1.0.2**：支持双语版本（中文/英文）和语言切换功能
- 💻 **Version 1.0.1**：修复 Windows 场景下文件相对路径和路径兼容性问题
- 🎉 **Version 1.0.0**：初始发布 ComfyUI-Lumi-Batcher，提供基础批量处理和多模态支持

&nbsp;

---

&nbsp;

## 🚀 如何安装

> 本地环境需要 Python3.10 以上版本

- 方式一：在 ComfyUI 的 CustomNode 目录，通过 git 命令拉取本项目，然后重启 ComfyUI 即可

```bash
git clone https://github.com/bytedance/comfyui-lumi-batcher.git
```

- 方式二：在 ComfyUI-Manager 中搜索`comfyui-lumi-batcher`，点击安装

安装完成后，在 ComfyUI 的 UI 面板中，默认在右上角位置，点击按钮，即可打开批量工具的界面。

&nbsp;

---

&nbsp;

## 🤝 获得帮助

**[新手帮助文档](https://bytedance.larkoffice.com/docx/LGLWdPIj8ooQyxxMAOQcWmR8nCh)**

<div style="margin-bottom:20px;font-weight: bold;">
  联系我们
</div>
<img alt="飞书群组" src="https://github.com/user-attachments/assets/b24c1e8c-dba8-47ed-9ad4-ed197f57301b" width="300" style="height:auto;" />
