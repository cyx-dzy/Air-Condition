# GitHub 上传准备说明

当前项目已经是一个本地 Git 仓库，并且已有首个提交：

```text
a5fe660 Build microgrid Python pipeline
```

## 当前缺少的条件

要把项目上传到 GitHub，需要满足下面任意一种条件：

1. 提供一个已经存在的 GitHub 仓库 SSH 地址，例如：

```text
git@github.com:username/repository.git
```

2. 在 Codex 的 GitHub 插件中连接一个可写账号，并提供目标仓库。
3. 安装并登录 GitHub CLI：

```bash
gh auth login
```

## 后续上传命令

拿到远程仓库地址后，可以执行：

```bash
git remote add origin git@github.com:username/repository.git
git branch -M main
git push -u origin main
```

## 安全提醒

不要把 GitHub 密码、SSH 密钥私钥、token 写入仓库文件。GitHub 推送通常使用 SSH key 或 personal access token，不使用账户密码。

