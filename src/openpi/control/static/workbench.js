"use strict";
const $ = (id) => document.getElementById(id),
  token = document.querySelector('meta[name="openpi-token"]').content,
  viewer = document.querySelector('meta[name="openpi-viewer"]').content,
  names = ["left", "main", "right"];
let hardwareMotion = false,
  exited = false,
  state = null,
  selectionDirty = false,
  paramsDirty = false,
  initialized = false,
  refKey = "",
  seq = 0,
  run = "",
  serverNow = 0,
  nowReceived = performance.now(),
  active = false,
  following = true,
  frozenNow = 0,
  lastTelemetrySuccess = 0,
  catalogKey = "";
const decisions = new Map(),
  acks = [],
  hidden = { ee: new Set(), joint: new Set() },
  palette = [
    "#2563eb",
    "#059669",
    "#ea580c",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#64748b",
    "#9333ea",
    "#e11d48",
    "#0284c7",
    "#65a30d",
    "#ca8a04",
    "#4f46e5",
    "#0f766e",
  ],
  cfgKeys = [
    "horizon",
    "execute_steps",
    "max_chunks",
    "control_hz",
    "velocity_mode",
    "position_tolerance_mm",
    "rotation_tolerance_rad",
  ];
function releaseViewer() {
  if (exited) return;
  exited = true;
  navigator.sendBeacon(
    "/api/viewer_leave",
    new Blob([JSON.stringify({ viewer, token })], { type: "application/json" }),
  );
}
window.addEventListener("pagehide", releaseViewer);
window.addEventListener("pageshow", (e) => {
  if (e.persisted) location.reload();
});
$("exitWorkbench").onclick = () => {
  releaseViewer();
  $("message").textContent = "页面已退出，所属模型正在关闭。";
  document
    .querySelectorAll("button,input,select")
    .forEach((e) => (e.disabled = true));
};
function config() {
  return Object.fromEntries(
    cfgKeys.map((k) => [
      k,
      k === "velocity_mode" ? $(k).value : Number($(k).value),
    ]),
  );
}
function valid(c) {
  return (
    ["horizon", "execute_steps", "max_chunks", "control_hz"].every((k) =>
      Number.isInteger(c[k]),
    ) &&
    c.execute_steps >= 1 &&
    c.execute_steps <= c.horizon &&
    c.horizon <= 30 &&
    c.max_chunks >= 1 &&
    c.max_chunks <= (state?.input_mode === "live" ? 1200 : 20) &&
    (state?.input_mode === "live"
      ? [1, 2, 5, 15, 30, 60, 90, 120]
      : [30]
    ).includes(c.control_hz) &&
    ["urdf", "collection"].includes(c.velocity_mode) &&
    c.position_tolerance_mm >= 0.1 &&
    c.position_tolerance_mm <= 5 &&
    c.rotation_tolerance_rad >= 0.0001 &&
    c.rotation_tolerance_rad <= 0.01
  );
}
function summary() {
  let c = config();
  $("configSummary").textContent = valid(c)
    ? `每次输出 30 步 · 显示 ${c.horizon} 步 · IK 前 ${c.execute_steps} 步 · 重复 ${c.max_chunks} 次`
    : "需要 1 ≤ 执行前缀 ≤ horizon ≤ 30，次数和频率须在当前模式范围内。";
  $("configSummary").classList.toggle("error", !valid(c));
  updateButtons();
}
async function post(action, payload = {}) {
  try {
    let r = await fetch("/api/" + action, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Replay-Token": token,
        },
        body: JSON.stringify({
          checkpoint_id: state?.selected_id,
          ...payload,
          viewer,
        }),
      }),
      j = await r.json();
    if (!r.ok) throw Error(j.error);
    return true;
  } catch (e) {
    $("message").textContent = e.message;
    $("message").classList.add("error");
    return false;
  }
}
function options(element, items, selected) {
  // Keep native menus mounted while state polling updates camera/daemon status.
  const changed =
    element.options.length !== items.length ||
    items.some(
      (item, index) =>
        element.options[index].value !== item.value ||
        element.options[index].text !== item.label,
    );
  if (!changed && document.activeElement === element) return;
  if (changed)
    element.replaceChildren(...items.map((x) => new Option(x.label, x.value)));
  if (element.value !== selected && items.some((x) => x.value === selected))
    element.value = selected;
}
function renderSteps(selected) {
  let task = state?.inventory.tasks.find((x) => x.id === $("task").value);
  options(
    $("model"),
    (task?.steps || []).map((x) => ({
      label: `${x.step.toLocaleString()} 步 · ${x.format}`,
      value: x.id,
    })),
    selected,
  );
}
function refreshChoices(s) {
  let key = JSON.stringify(s.inventory.tasks);
  if (key !== catalogKey) {
    let task = selectionDirty ? $("task").value : s.task,
      id = selectionDirty ? $("model").value : s.selected_id;
    options(
      $("task"),
      s.inventory.tasks.map((x) => ({ label: x.id, value: x.id })),
      task,
    );
    state = s;
    renderSteps(id);
    catalogKey = key;
    if (initialized && $("model").value !== s.selected_id)
      selectionDirty = true;
  }
  $("catalogStatus").textContent =
    `${s.inventory.tasks.length} 个训练任务 · ${s.inventory.tasks.reduce((a, x) => a + x.steps.length, 0)} 份 checkpoint` +
    (s.inventory.unavailable.length
      ? ` · ${s.inventory.unavailable.length} 个目录暂不可用`
      : "");
  $("catalogStatus").title =
    s.inventory.root +
    "\n" +
    s.inventory.unavailable.map((x) => x.id + ": " + x.reason).join("\n");
}
function updateButtons() {
  if (!state || exited) return;
  let busy = state.busy,
    dirty =
      selectionDirty ||
      paramsDirty ||
      (state.input_mode === "live" && referenceDirty);
  for (let id of [
    "task",
    "model",
    "episode",
    "horizon",
    "execute_steps",
    "max_chunks",
    "velocity_mode",
    "position_tolerance_mm",
    "rotation_tolerance_rad",
    "applyTask",
    "applyEpisode",
    "resetConfig",
    "refreshCatalog",
  ])
    $(id).disabled = busy;
  $("applyConfig").disabled = busy || !valid(config());
  $("control_hz").disabled = busy || state.input_mode !== "live";
  for (let id of ["position_tolerance_mm", "rotation_tolerance_rad"])
    $(id).disabled = busy || state.input_mode !== "live";
  let loaded = state.server?.selected_match;
  $("startServer").disabled =
    busy || selectionDirty || !state.policy_ready || loaded;
  $("startServer").textContent =
    state.server?.phase === "loading"
      ? "加载中…"
      : loaded
        ? "Server 已启动"
        : "启动 server";
  $("stopServer").disabled = busy || state.server?.phase !== "ready";
  $("shadow").disabled = busy || dirty || !state.replay_ready || !loaded;
  let live = state.live || {},
    isLive = state.input_mode === "live";
  for (let id of [
    "input_mode",
    "reference",
    "applyReference",
    "skip",
    "aligned",
    "plan",
    "prepare",
    "connectLive",
  ])
    $(id).disabled = busy;
  $("input_mode").querySelector('option[value="live"]').disabled =
    !state.capabilities.live_cameras;
  $("connectLive").disabled = busy || !state.capabilities.live_cameras;
  $("reference").disabled = busy || !isLive || !live.references?.length;
  $("applyReference").disabled = $("reference").disabled;
  $("plan").disabled =
    busy || dirty || !live.ready || !isLive || live.alignment_skipped;
  $("aligned").disabled =
    busy || dirty || !live.ready || !isLive || live.alignment_skipped;
  $("prepare").disabled =
    busy ||
    dirty ||
    !isLive ||
    !live.ready ||
    !live.plan_valid ||
    !live.aligned ||
    live.alignment_skipped;
  $("policy").disabled =
    busy ||
    dirty ||
    !isLive ||
    !live.ready ||
    !loaded ||
    !(live.aligned || live.alignment_skipped);
  $("applyTask").disabled = busy || !$("model").value;
  $("applyEpisode").disabled = busy || !state.samples.length;
  $("applyTask").textContent = selectionDirty ? "应用选择 *" : "应用选择";
  $("applyConfig").textContent = paramsDirty
    ? "应用推理参数 *"
    : "应用推理参数";
}
$("task").onchange = () => {
  renderSteps();
  selectionDirty = true;
  updateButtons();
};
$("model").onchange = () => {
  selectionDirty = true;
  updateButtons();
};
$("episode").onchange = () => {
  selectionDirty = true;
  updateButtons();
};
async function applyTask() {
  if (
    await post("select", {
      checkpoint_id: $("model").value,
      sample:
        $("model").value === state.selected_id
          ? Number($("episode").value || 0)
          : 0,
    })
  ) {
    selectionDirty = false;
    referenceDirty = false;
    await pollState();
  }
}
$("applyTask").onclick = applyTask;
$("applyEpisode").onclick = applyTask;
$("refreshCatalog").onclick = async () => {
  await fetch("/api/catalog");
  await pollState();
};
$("startServer").onclick = () =>
  post("server_start", { checkpoint_id: state.selected_id });
$("stopServer").onclick = () => post("server_stop");
$("stop").onclick = () => post("stop");
$("shadow").onclick = () =>
  post("shadow", {
    checkpoint_id: state.selected_id,
    settings_revision: state.settings_revision,
  });
for (let k of cfgKeys)
  $(k).oninput = () => {
    paramsDirty = true;
    summary();
  };
$("applyConfig").onclick = async () => {
  if (valid(config()) && (await post("configure", config()))) {
    paramsDirty = false;
    await pollState();
  }
};
$("resetConfig").onclick = () => {
  let c = {
    horizon: 30,
    execute_steps: 15,
    max_chunks: state?.input_mode === "live" ? 120 : 1,
    control_hz: 30,
    velocity_mode: state?.input_mode === "live" ? "collection" : "urdf",
    position_tolerance_mm: state?.input_mode === "live" ? 5 : 0.1,
    rotation_tolerance_rad: state?.input_mode === "live" ? 0.01 : 0.0001,
  };
  cfgKeys.forEach((k) => ($(k).value = c[k]));
  paramsDirty = true;
  summary();
};
$("toggleSidebar").onclick = () => $("sidebar").classList.toggle("open");
const phases = {
  preview: "待机",
  planning: "起点预检中",
  planned: "预检通过",
  preparing: "回位中",
  at_start: "起点就绪",
  policy: "真机策略运行",
  shadow: "只读推理",
  server_starting: "加载模型",
  server_ready: "模型就绪",
  complete: "本轮结束",
  rejected: "本轮中止",
};
async function pollState() {
  try {
    let r = await fetch("/api/state?viewer=" + viewer);
    if (!r.ok) throw Error("状态服务不可用");
    let s = await r.json();
    state = s;
    refreshChoices(s);
    if (!initialized) {
      cfgKeys.forEach((k) => ($(k).value = s.settings[k]));
      initialized = true;
      summary();
    }
    if (!selectionDirty) {
      $("task").value = s.task;
      renderSteps(s.selected_id);
      options(
        $("episode"),
        s.samples.map((x, i) => ({
          label: x.task + " · " + x.original_id,
          value: String(i),
        })),
        String(s.sample_index),
      );
    }
    if (!paramsDirty)
      cfgKeys.forEach((k) => {
        const element = $(k);
        if (
          document.activeElement !== element &&
          element.value !== String(s.settings[k])
        )
          element.value = s.settings[k];
      });
    let srv = s.server;
    $("serverStatus").textContent =
      srv.phase === "ready" ? `已加载 · PID ${srv.pid}` : srv.message;
    $("serverStatus").title =
      (srv.checkpoint || "") +
      (srv.model_sha256 ? "\nSHA256 " + srv.model_sha256 : "");
    $("serverStatus").classList.toggle("ready", srv.selected_match);
    $("serverStatus").classList.toggle("error", srv.phase === "error");
    $("serverBadge").textContent = srv.selected_match
      ? "Server 已加载"
      : srv.phase === "loading"
        ? "Server 加载中"
        : "Server 未就绪";
    $("connection").textContent = "OpenPI 已连接";
    $("connection").classList.remove("error");
    $("phase").textContent = phases[s.phase] || s.phase;
    $("message").textContent = s.message;
    $("message").title = s.message;
    $("message").classList.toggle("error", s.phase === "rejected");
    $("modelReady").textContent = s.policy_ready
      ? "文件齐全 · 启动时核验 SHA 并加载模型"
      : "所选 checkpoint 当前不可用";
    $("subtitle").textContent =
      (s.task || "尚无 checkpoint") +
      (s.checkpoint_step != null
        ? " · step " + s.checkpoint_step.toLocaleString()
        : "");
    renderLive(s);
    $("sampleInfo").textContent = s.sample
      ? `${s.sample.task} · 保存帧 ${s.sample.index} · target support`
      : "该 checkpoint 没有此后端可用的保存回放。";
    let key =
      s.input_mode +
      ":" +
      s.selected_id +
      ":" +
      (s.input_mode === "live" ? s.live?.reference?.sha256 : s.sample_index);
    if (refKey !== key) {
      refKey = key;
      for (let n of names)
        $("live-" + n).src = "/camera/" + n + "?v=" + encodeURIComponent(key);
    }
    let c = s.settings;
    $("horizonStat").textContent = c.horizon;
    $("horizonTime").textContent = `步 · ${(c.horizon / 30).toFixed(2)} s`;
    $("executeStat").textContent = c.execute_steps;
    $("executeTime").textContent =
      s.input_mode === "live" ? "步 · 现场 IK" : "步 · 离线 IK";
    $("hzStat").textContent = c.control_hz;
    document
      .querySelectorAll(".age")
      .forEach(
        (x) =>
          (x.textContent =
            s.input_mode === "live"
              ? s.live?.camera_age_s != null
                ? Math.round(s.live.camera_age_s * 1000) + " ms"
                : "等待现场画面"
              : s.sample
                ? "保存帧"
                : "无输入"),
      );
    $("progress").textContent = s.busy
      ? `运行中 · 已完成 ${s.completed_chunks} / ${c.max_chunks} 次 · ${s.hardware_motion ? "正在发送运动" : "未发送运动"}`
      : `待机 · 上轮完成 ${s.completed_chunks} 次 · 下轮 ${c.max_chunks} 次 · ${s.hardware_motion ? "正在发送运动" : "未发送运动"}`;
    $("logpath").textContent = s.policy_output || "日志将在推理后生成";
    updateButtons();
  } catch (e) {
    $("connection").textContent = "连接中断";
    $("connection").classList.add("error");
    $("shadow").disabled = true;
    $("policy").disabled = true;
    $("prepare").disabled = true;
    $("message").textContent = e.message;
  }
}

let referenceKey = "",
  referenceDirty = false;
function overlay() {
  let live = state?.input_mode === "live",
    skip = $("skip").checked,
    opacity = Number($("opacity").value) / 100;
  $("opacityLabel").textContent = Math.round(opacity * 100) + "%";
  $("alignment").classList.toggle("hidden", skip);
  for (let n of names) {
    $("ref-" + n).style.opacity = live && !skip ? opacity : 0;
    $("ref-" + n).style.display = live && !skip ? "block" : "none";
  }
}
$("opacity").oninput = overlay;
$("input_mode").onchange = async () => {
  if (await post("input_mode", { mode: $("input_mode").value })) {
    paramsDirty = false;
    selectionDirty = false;
    referenceDirty = false;
    await pollState();
    summary();
  }
};
$("connectLive").onclick = () => post("connect_live");
$("reference").onchange = () => {
  referenceDirty = true;
  updateButtons();
};
$("applyReference").onclick = async () => {
  if (await post("reference", { index: Number($("reference").value) })) {
    referenceDirty = false;
    await pollState();
  }
};
$("skip").onchange = () => {
  overlay();
  post("skip_alignment", { skip: $("skip").checked });
};
$("aligned").onchange = () => post("align", { aligned: $("aligned").checked });
$("plan").onclick = () => post("plan");
$("prepare").onclick = () =>
  post("prepare", { plan_id: state?.live?.plan?.plan_id });
$("policy").onclick = () =>
  post("policy", { settings_revision: state.settings_revision });
function renderLive(s) {
  let live = s.live || {},
    isLive = s.input_mode === "live";
  if (
    document.activeElement !== $("input_mode") &&
    $("input_mode").value !== s.input_mode
  )
    $("input_mode").value = s.input_mode;
  $("savedInput").classList.toggle("hidden", isLive);
  $("liveInput").classList.toggle("hidden", !isLive);
  $("connectLive").classList.toggle("hidden", !isLive);
  $("policy").classList.toggle("hidden", !isLive);
  $("inputDescription").textContent = isLive
    ? "现场观测 · 闭环策略"
    : "保存回放 · 只读推理";
  $("inputBadge").textContent = isLive ? "现场实时观测" : "保存回放";
  $("liveStatus").textContent = isLive
    ? live.error ||
      (!live.ready
        ? "正在连接现场观测…"
        : `三相机新鲜 · 同步偏差 ${Math.round((live.camera_skew_s || 0) * 1000)} ms · daemon ${live.daemon?.state || "unknown"}`)
    : "";
  $("liveStatus").classList.toggle("error", !!live.error);
  if (!referenceDirty)
    options(
      $("reference"),
      (live.references || []).map((x, i) => ({
        value: String(i),
        label: x.task + " · " + x.original_id,
      })),
      String(live.reference_index ?? 0),
    );
  $("skip").checked = !!live.alignment_skipped;
  $("aligned").checked = !!live.aligned;
  $("planInfo").textContent = live.plan
    ? live.plan_valid
      ? `预检通过 · 预计 ${live.plan.execution_duration_s.toFixed(2)} 秒；回位需要单独点击。`
      : "预检已过期或反馈变化，请重新预检。"
    : "叠加训练示范首帧；回位目标使用原始实测关节。";
  let key = s.selected_id + ":" + (live.reference?.sha256 || "");
  if (key !== referenceKey) {
    referenceKey = key;
    for (let n of names)
      $("ref-" + n).src = "/reference/" + n + "?v=" + encodeURIComponent(key);
  }
  $("max_chunks").max = isLive ? "1200" : "20";
  document
    .querySelectorAll(".camerafoot>span:first-child")
    .forEach(
      (x) => (x.textContent = isLive ? "现场 / 训练示范叠图" : "保存回放输入"),
    );
  overlay();
}
for (let n of names) {
  const image = $("live-" + n);
  const update = () => {
    if (exited) return;
    setTimeout(() => {
      if (!exited && state?.input_mode === "live")
        image.src = "/camera/" + n + "?t=" + Date.now();
    }, 150);
  };
  image.onload = update;
  image.onerror = () => {
    if (!exited)
      setTimeout(() => {
        if (!exited && state?.input_mode === "live")
          image.src = "/camera/" + n + "?t=" + Date.now();
      }, 800);
  };
}
function ingest(rows) {
  for (let e of rows) {
    if (e.kind === "start") hardwareMotion = e.hardware_motion;
    if (["decision", "prediction", "ik", "submit"].includes(e.kind)) {
      let d = decisions.get(e.index) || { index: e.index };
      if (e.kind === "decision") d.decision = e.t;
      else if (e.kind === "prediction")
        Object.assign(d, {
          prediction: e.t,
          actions: e.actions,
          horizon: e.horizon,
          execute: e.execute_steps,
          latency: e.inference_s,
        });
      else if (e.kind === "ik") {
        d.ik = e.points;
        d.stepPeriod = e.step_period_s || 1 / 30;
      } else {
        d.submit = e.t;
        d.hz = e.control_hz;
        d.stepPeriod = e.step_period_s || 1 / 30;
      }
      decisions.set(e.index, d);
    } else if (e.kind === "ack") acks.push(e);
  }
  while (acks.length > 9000) acks.shift();
  let limit = serverNow - 125;
  for (let [k, d] of decisions)
    if ((d.decision ?? 0) < limit) decisions.delete(k);
}
async function pollTelemetry() {
  try {
    let r = await fetch(
      `/api/telemetry?after=${seq}&run=${encodeURIComponent(run)}`,
    );
    if (!r.ok) throw Error("telemetry");
    let j = await r.json();
    if (j.reset) {
      decisions.clear();
      acks.length = 0;
      seq = 0;
    }
    run = j.run || "";
    serverNow = j.now;
    active = j.active;
    nowReceived = performance.now();
    lastTelemetrySuccess = performance.now();
    ingest(j.events);
    seq = j.seq;
    $("decisions").textContent = decisions.size
      ? Math.max(...decisions.keys()) + 1
      : 0;
    let ds = Array.from(decisions.values()),
      latest = ds.at(-1);
    $("inferenceInfo").textContent = latest?.actions
      ? `D${latest.index + 1} · 推理 ${(latest.latency * 1000).toFixed(0)} ms · 原始 30 / 采用 ${latest.horizon} 步`
      : "等待模型推理";
    let recent = acks.slice(-121);
    let hz =
      recent.length > 2
        ? (recent.length - 1) / (recent.at(-1).t - recent[0].t)
        : 0;
    $("actualHz").textContent =
      hz > 0 ? `Hz · 底层 ACK ${hz.toFixed(1)} Hz` : "Hz · 模型时间基准";
    $("ackInfo").textContent = acks.length
      ? `收到 ${acks.length} 条 ACK · 含等待期间保持指令`
      : hardwareMotion === false
        ? "只读推理 · 未发送运动指令"
        : "等待离线 IK";
    $("eeEmpty").classList.toggle(
      "hidden",
      ds.some((d) => d.actions),
    );
    $("jointEmpty").classList.toggle(
      "hidden",
      ds.some((d) => d.ik) || acks.length > 0,
    );
  } catch (e) {
    $("inferenceInfo").textContent = "遥测连接中断 · 曲线已冻结";
  }
}
let groups = {
  ee: {
    position: [0, 1, 2, 7, 8, 9],
    rotation: [3, 4, 5, 10, 11, 12],
    gripper: [6, 13],
  },
  joint: {
    left: [0, 1, 2, 3, 4, 5],
    right: [7, 8, 9, 10, 11, 12],
    gripper: [6, 13],
  },
};
function dims(kind) {
  return groups[kind][$(kind === "ee" ? "eeGroup" : "jointGroup").value];
}
function dimName(kind, i) {
  let side = i < 7 ? "L" : "R",
    j = i % 7;
  return (
    side +
    " " +
    (kind === "ee"
      ? ["Δx", "Δy", "Δz", "rx", "ry", "rz", "Grip"][j]
      : ["J1", "J2", "J3", "J4", "J5", "J6", "Grip"][j])
  );
}
function legend(kind) {
  $(kind + "Legend").innerHTML = "";
  dims(kind).forEach((i, k) => {
    let b = document.createElement("button");
    b.innerHTML = `<i class="swatch" style="border-color:${palette[i]};border-top-style:solid"></i>${dimName(kind, i)}`;
    b.classList.toggle("off", hidden[kind].has(i));
    b.onclick = () => {
      hidden[kind].has(i) ? hidden[kind].delete(i) : hidden[kind].add(i);
      legend(kind);
    };
    $(kind + "Legend").appendChild(b);
  });
}
for (let kind of ["ee", "joint"]) {
  legend(kind);
  $(kind === "ee" ? "eeGroup" : "jointGroup").onchange = () => legend(kind);
}
$("follow").onclick = () => {
  following = !following;
  if (!following) frozenNow = plotNow();
  $("follow").textContent = following ? "暂停跟随" : "恢复实时跟随";
};
function plotNow() {
  if (!following) return frozenNow;
  return (
    serverNow +
    (active && performance.now() - lastTelemetrySuccess < 1500
      ? (performance.now() - nowReceived) / 1000
      : 0)
  );
}
const chartPoints = {};
function draw(kind) {
  let canvas = $(kind + "Chart"),
    rect = canvas.getBoundingClientRect(),
    w = rect.width,
    h = rect.height,
    dpr = window.devicePixelRatio || 1;
  if (w < 60 || h < 50) return;
  if (
    canvas.width !== Math.round(w * dpr) ||
    canvas.height !== Math.round(h * dpr)
  ) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  let ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  let tNow = plotNow(),
    windowSpan = +$("window").value,
    end = Math.max(2, tNow + 1.5),
    start = Math.max(0, end - windowSpan),
    span = end - start,
    L = w < 300 ? 38 : 48,
    R = w - 12,
    T = 28,
    B = h - 22,
    X = (t) => L + ((t - start) / span) * (R - L),
    series = [],
    selected = dims(kind).filter((i) => !hidden[kind].has(i));
  for (let d of decisions.values()) {
    if (d.decision == null) continue;
    if (kind === "ee" && d.actions) {
      let base = d.decision;
      for (let i of selected)
        series.push({
          i,
          pts: d.actions
            .slice(0, d.horizon)
            .map((v, k) => [base + (k + 1) / 30, v[i]]),
          d,
          mode: "ee",
        });
    } else if (kind === "joint" && d.ik) {
      let base = d.submit ?? d.prediction;
      for (let i of selected)
        series.push({
          i,
          pts: d.ik.map((v, k) => [
            base + 1 / 30 + k * (d.stepPeriod || 1 / 30),
            v[i],
          ]),
          d,
          mode: "ik",
        });
    }
  }
  if (kind === "joint")
    for (let i of selected) {
      series.push({
        i,
        pts: acks.map((a) => [a.t, a.accepted[i]]),
        mode: "accepted",
      });
      series.push({
        i,
        pts: acks.map((a) => [a.t, a.measured[i]]),
        mode: "measured",
      });
    }
  let vals = series
    .flatMap((s) =>
      s.pts.filter((p) => p[0] >= start && p[0] <= end).map((p) => p[1]),
    )
    .filter(Number.isFinite);
  let min = vals.length ? Math.min(...vals) : 0,
    max = vals.length ? Math.max(...vals) : 1;
  if (max - min < 0.001) {
    min -= 0.005;
    max += 0.005;
  }
  let pad = (max - min) * 0.12;
  min -= pad;
  max += pad;
  let Y = (v) => B - ((v - min) / (max - min)) * (B - T);
  ctx.font = (w < 300 ? "9" : "11") + "px Inter,system-ui";
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#ececec";
  ctx.fillStyle = "#737373";
  for (let k = 0; k <= 3; k++) {
    let v = min + ((max - min) * k) / 3,
      y = Y(v);
    ctx.beginPath();
    ctx.moveTo(L, y);
    ctx.lineTo(R, y);
    ctx.stroke();
    ctx.fillText(v.toFixed(Math.abs(v) < 0.1 ? 3 : 2), 3, y + 4);
  }
  let ticks = w < 300 ? 3 : 5;
  for (let k = 0; k <= ticks; k++) {
    let t = start + (span * k) / ticks,
      x = X(t);
    ctx.beginPath();
    ctx.moveTo(x, T);
    ctx.lineTo(x, B);
    ctx.stroke();
    ctx.fillText(t.toFixed(1) + " s", x - 13, h - 7);
  }
  ctx.save();
  ctx.beginPath();
  ctx.rect(L, 0, R - L, B);
  ctx.clip();
  for (let d of decisions.values()) {
    let t = d.decision;
    if (
      t == null ||
      t > end ||
      t +
        (kind === "ee"
          ? (d.horizon || 30) / 30
          : (d.execute || 15) * (d.stepPeriod || 1 / 30)) <
        start
    )
      continue;
    let hi =
        kind === "ee"
          ? (d.horizon || 30) / 30
          : (d.execute || 15) * (d.stepPeriod || 1 / 30),
      ex =
        (d.execute || state?.settings.execute_steps || 15) *
        (kind === "ee" ? 1 / 30 : d.stepPeriod || 1 / 30);
    ctx.fillStyle = "#f5f5f5";
    ctx.fillRect(X(t), T, X(t + hi) - X(t), B - T);
    ctx.fillStyle = "#e5e5e5";
    ctx.fillRect(X(t), T, X(t + ex) - X(t), 4);
    ctx.strokeStyle = "#a3a3a3";
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(X(t), T);
    ctx.lineTo(X(t), B);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#525252";
    ctx.fillText("D" + (d.index + 1), X(t) + 3, 12 + (d.index % 2) * 12);
    if (d === Array.from(decisions.values()).at(-1)) {
      ctx.fillText(
        (kind === "ee" ? "H " + (d.horizon || 30) : "IK " + (d.execute || 15)) +
          " · " +
          hi.toFixed(2) +
          "s",
        X(t) + 4,
        B - 6,
      );
    }
  }
  ctx.restore();
  ctx.save();
  ctx.beginPath();
  ctx.rect(L, T, R - L, B - T);
  ctx.clip();
  let hit = [];
  for (let s of series) {
    let k = dims(kind).indexOf(s.i),
      pts = s.pts.filter((p) => p[0] >= start - 0.1 && p[0] <= end + 0.1);
    ctx.strokeStyle = palette[s.i];
    ctx.globalAlpha = s.mode === "measured" ? 0.4 : s.mode === "ik" ? 0.85 : 1;
    ctx.lineWidth = s.mode === "accepted" ? 2 : 1.5;
    ctx.setLineDash(s.mode === "ik" ? [5, 3] : []);
    let prev = null;
    for (let p of pts) {
      if (prev) {
        if (s.mode === "ee")
          ctx.setLineDash(
            p[0] <= s.d.decision + s.d.execute / 30 ? [] : [3, 4],
          );
        ctx.beginPath();
        ctx.moveTo(X(prev[0]), Y(prev[1]));
        ctx.lineTo(X(p[0]), Y(p[1]));
        ctx.stroke();
      }
      prev = p;
      if (s.mode !== "measured")
        hit.push({
          x: X(p[0]),
          y: Y(p[1]),
          t: p[0],
          v: p[1],
          i: s.i,
          mode: s.mode,
        });
    }
  }
  ctx.globalAlpha = 1;
  ctx.setLineDash([]);
  if (tNow >= start && tNow <= end) {
    ctx.strokeStyle = "#0a0a0a";
    ctx.beginPath();
    ctx.moveTo(X(tNow), T);
    ctx.lineTo(X(tNow), B);
    ctx.stroke();
  }
  ctx.restore();
  ctx.fillStyle = "#171717";
  if (tNow >= start && tNow <= end)
    ctx.fillText("NOW", Math.min(R - 25, X(tNow) + 4), h - 7);
  chartPoints[kind] = hit;
}
for (let kind of ["ee", "joint"]) {
  $(kind + "Chart").onpointermove = (e) => {
    let r = e.currentTarget.getBoundingClientRect(),
      x = e.clientX - r.left,
      y = e.clientY - r.top,
      points = chartPoints[kind] || [],
      near = points.reduce(
        (a, p) =>
          !a || Math.hypot(p.x - x, p.y - y) < Math.hypot(a.x - x, a.y - y)
            ? p
            : a,
        null,
      );
    if (!near || Math.hypot(near.x - x, near.y - y) > 36) {
      $("tooltip").classList.add("hidden");
      return;
    }
    let unit =
      near.i % 7 === 6
        ? "0–1"
        : kind === "ee" && $("eeGroup").value === "position"
          ? "m"
          : "rad";
    $("tooltip").textContent =
      `${dimName(kind, near.i)}  ${near.v.toFixed(5)} ${unit}\nt = ${near.t.toFixed(3)} s · ${near.mode}`;
    $("tooltip").style.borderLeft = "3px solid " + palette[near.i];
    $("tooltip").style.left = Math.min(e.clientX + 12, innerWidth - 260) + "px";
    $("tooltip").style.top = Math.max(8, e.clientY - 60) + "px";
    $("tooltip").classList.remove("hidden");
  };
  $(kind + "Chart").onpointerleave = () => $("tooltip").classList.add("hidden");
}
let lastDraw = 0;
function animate(t) {
  if (t - lastDraw > 66) {
    draw("ee");
    draw("joint");
    lastDraw = t;
  }
  requestAnimationFrame(animate);
}
async function loop() {
  if (exited) return;
  await pollState();
  if (exited) return;
  await pollTelemetry();
  if (!exited) setTimeout(loop, 250);
}
loop();
requestAnimationFrame(animate);
