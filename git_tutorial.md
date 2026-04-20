# 0) 進入專案
cd repo

# 1) 更新 main
git checkout main
git pull origin main

# 2) 從 main 切出功能分支
git switch -c feature/my-task

# 3) 開發 → 提交
# ...編輯檔案...
git status
git add -A
git commit -m "feat: implement my-task"

# 4)（可選）接上最新 main，保持乾淨歷史
git fetch origin
git rebase origin/main    # 若有衝突：修 → git add → git rebase --continue

# 5) 推上遠端
git push -u origin feature/my-task

# 6) 到 GitHub/GitLab 開 PR（base: main ← compare: feature/my-task）
<!--  通過後 Merge（Squash/FF/Regular 依團隊設定，單人隨意） -->

# 7) 合併完成後清理分支
git push origin --delete feature/my-task   # 刪遠端分支(merge PR的時候，可以用按鈕刪除)
git checkout main
git pull origin main
git branch -d feature/my-task              # 刪本地分支


### 復原版本
在 GitHub 上點進 Commits 頁面，找到想要復原的 commit
在專案首頁，靠近上方會有幾個分頁：Code,IssuesPull ,requests, Actions

在 Code 頁面下方（檔案列表上方），你會看到 branches、tags、還有 commits 數字。
點擊 commits 數字（通常會顯示像 34 commits）。
就會進入 Commits 歷史頁面，可以看到所有提交記錄。

- 點每個 commit 的 <> 按鈕（Browse files），查看那個版本完整檔案。
- 或者點 Revert（在 Pull Request/合併過的 commit 旁邊會有）來建立一個反向 commit。