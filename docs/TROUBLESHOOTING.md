# 故障排查指南

## 视频下载 403 错误

### 问题原因

抖音视频受到防盗链保护，403 错误通常是因为：

1. **URL 不是视频直链**：传入了抖音网页链接而不是视频文件直链
2. **请求头不正确**：缺少必要的 headers 或顺序不对
3. **IP 或 Cookies 限制**：抖音可能限制了某些 IP 或需要登录状态

### 解决方案

#### 1. 使用视频直链 URL

**重要**：必须使用视频文件直链，而不是抖音网页链接。

```
❌ 错误：https://www.douyin.com/video/7234567890123456789
✅ 正确：https://...aweme.snssdk.com/aweme/v1/play/...
```

#### 2. 解析抖音链接获取直链

参考 Node.js 代码中的 `parseDouyinUrl` 函数：

```javascript
// 调用解析服务获取视频直链
const response = await fetch(`http://nas.linutone.com:8002/api/work/info?url=${encodeURIComponent(url)}`);
const videoUrl = response.data.videoUrl;

// 然后使用 videoUrl 调用转写 API
```

#### 3. 检查日志

启用详细日志以诊断问题：

```bash
# 查看容器日志
docker logs -f sensevoice-http

# 重新构建时启用调试日志
docker-compose down
docker-compose build
docker-compose up -d
```

日志会显示：
- 使用的 headers
- 响应状态码
- 响应内容

### 测试步骤

1. **验证 URL 格式**：
   ```bash
   # 检查 URL 是否包含视频文件扩展名或 /aweme/
   echo $VIDEO_URL | grep -E '\.(mp4|webm|flv)|/aweme/'
   ```

2. **使用 curl 测试**：
   ```bash
   curl -I "$VIDEO_URL" \
     -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
     -H "Referer: https://www.douyin.com/"
   ```

3. **查看响应状态码**：
   - 200: 成功
   - 403: 防盗链失败
   - 404: URL 无效

### 常见错误信息

#### "请提供视频直链 URL"

**原因**：传入了抖音网页链接而不是视频文件链接

**解决**：先调用解析服务获取视频直链，参考上面的 Node.js 代码

#### "下载视频失败，状态码: 403"

**原因**：防盗链检查失败

**可能原因**：
1. Headers 不正确
2. IP 被限制
3. 需要 Cookies
4. URL 已过期

**解决**：
1. 检查日志中的请求头是否正确
2. 尝试使用代理
3. 添加有效的 Cookies
4. 重新获取视频 URL

### 调试技巧

1. **启用详细日志**：
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **使用 curl 验证**：
   ```bash
   curl -v "$VIDEO_URL" \
     -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
     -H "Referer: https://www.douyin.com/" \
     --output test.mp4
   ```

3. **检查响应头**：
   查看 `Content-Type` 是否为 `video/mp4` 或类似的视频类型

### 参考链接

- [Docker 部署指南](DOCKER.md)
- [API 使用文档](API_USAGE.md)
- [抖音视频下载说明](DOUYIN_DOWNLOAD.md)

