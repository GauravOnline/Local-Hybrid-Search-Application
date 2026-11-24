/* All requests stay on the local FastAPI server. */
const byId = id => document.getElementById(id);

function element(tag, text, className) {
    const node = document.createElement(tag);
    node.textContent = text;
    if (className) node.className = className;
    return node;
}

async function request(path, body) {
    const options = body === undefined ? {} : {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
    };
    const response = await fetch(path, options);
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Request failed.");
    return data;
}

function setBusy(busy) {
    document.querySelectorAll("button").forEach(button => button.disabled = busy);
}

async function refreshStatus() {
    const status = await request("/api/status");
    byId("file-count").textContent = status.files;
    byId("chunk-count").textContent = status.chunks;
    byId("vector-count").textContent = status.vectors;
    byId("storage").textContent = status.storage_bytes >= 1048576
        ? (status.storage_bytes / 1048576).toFixed(2) + " MB"
        : (status.storage_bytes / 1024).toFixed(1) + " KB";
    const select = byId("filter-folder");
    const selected = select.value;
    select.replaceChildren(new Option("All folders", ""));
    byId("folders").replaceChildren();
    for (const folder of status.folders) {
        select.add(new Option(folder, folder));
        const item = element("div", "", "folder-item");
        item.append(element("span", folder));
        const refresh = element("button", "Refresh");
        refresh.addEventListener("click", () => indexFolder(folder));
        item.append(refresh);
        byId("folders").append(item);
    }
    select.value = status.folders.includes(selected) ? selected : "";
}

async function indexFolder(folder) {
    setBusy(true);
    const panel = byId("index-report");
    panel.replaceChildren(element("div", "Indexing files… First run may take a while.", "report"));
    try {
        const result = await request("/api/index", {folder});
        const summary = element("div",
            `Scan complete · ${result.seconds}s\nAdded: ${result.added} · Updated: ${result.updated} · Unchanged: ${result.unchanged} · Removed: ${result.removed} · Errors: ${result.errors.length}`,
            "report");
        summary.style.whiteSpace = "pre-line";
        for (const issue of [...result.errors, ...result.warnings]) {
            summary.append(element("p", `${issue.path}: ${issue.message}`, result.errors.includes(issue) ? "error" : ""));
        }
        panel.replaceChildren(summary);
        await refreshStatus();
        byId("results").replaceChildren(element("div", "Index refreshed. Run a new search.", "empty"));
        byId("search-status").textContent = "Ready";
    } catch (error) {
        panel.replaceChildren(element("div", error.message, "report error"));
    } finally {
        setBusy(false);
    }
}

function renderResults(data) {
    const container = byId("results");
    container.replaceChildren();
    byId("search-status").textContent = `${data.results.length} results · ${data.mode} · ${data.latency_ms.toFixed(1)} ms`;
    if (!data.results.length) {
        container.append(element("div", "No matching files. Try another query or filter.", "empty"));
        return;
    }
    for (const item of data.results) {
        const card = element("article", "", "result");
        const top = element("div", "", "result-top");
        const title = element("div", "");
        title.append(element("h3", `#${item.rank}  ${item.name}`), element("div", item.path, "path"));
        const open = element("button", "Open file");
        open.addEventListener("click", async () => {
            try { await request(`/api/files/${item.id}/open`, {}); }
            catch (error) { byId("search-status").textContent = error.message; }
        });
        top.append(title, open);
        const metrics = element("div", "", "metrics");
        for (const label of [item.extension.toUpperCase(), `${(item.size_bytes / 1024).toFixed(1)} KB`,
            `Score: ${item.score.toFixed(4)}`, item.keyword_rank ? `Keyword #${item.keyword_rank}` : "",
            item.semantic_rank ? `Semantic #${item.semantic_rank}` : ""]) {
            if (label) metrics.append(element("span", label));
        }
        card.append(top, element("p", item.snippet, "snippet"), metrics);
        container.append(card);
    }
}

byId("index-form").addEventListener("submit", event => {
    event.preventDefault();
    indexFolder(byId("folder").value.trim());
});

byId("search-form").addEventListener("submit", async event => {
    event.preventDefault();
    setBusy(true);
    byId("search-status").textContent = "Searching…";
    try {
        renderResults(await request("/api/search", {
            query: byId("query").value.trim(),
            mode: document.querySelector("input[name=mode]:checked").value,
            extension: byId("type").value,
            folder: byId("filter-folder").value,
            limit: Number(byId("limit").value)
        }));
    } catch (error) {
        byId("search-status").textContent = "Search failed";
        byId("results").replaceChildren(element("div", error.message, "empty error"));
    } finally {
        setBusy(false);
    }
});

refreshStatus().catch(error => byId("search-status").textContent = error.message);
