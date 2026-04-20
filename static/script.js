/**
 * ReviewGuard AI — Frontend Logic
 * Handles review analysis, result rendering, star rating, and chart display.
 */

(() => {
    "use strict";

    // ── DOM References ───────────────────────────────────────────────────
    const reviewInput      = document.getElementById("review-input");
    const charCount        = document.getElementById("char-count");
    const analyzeBtn       = document.getElementById("analyze-btn");
    const loader           = document.getElementById("loader");
    const resultsSection   = document.getElementById("results-section");

    // Result elements
    const sentimentIcon    = document.getElementById("sentiment-icon");
    const sentimentLabel   = document.getElementById("sentiment-label");
    const sentimentBar     = document.getElementById("sentiment-bar");
    const sentimentConf    = document.getElementById("sentiment-confidence");
    const sentimentReason  = document.getElementById("sentiment-reason");

    const fakeIcon         = document.getElementById("fake-icon");
    const fakeLabel        = document.getElementById("fake-label");
    const fakeBar          = document.getElementById("fake-bar");
    const fakeConf         = document.getElementById("fake-confidence");
    const fakeReason       = document.getElementById("fake-reason");

    const starDisplay      = document.getElementById("star-display");
    const starValue        = document.getElementById("star-value");

    const chartCanvas      = document.getElementById("confidence-chart");

    let chartInstance = null;

    // ── Character Counter ────────────────────────────────────────────────
    reviewInput.addEventListener("input", () => {
        const len = reviewInput.value.length;
        charCount.textContent = `${len} / 2000`;
    });

    // ── Analyze Button ───────────────────────────────────────────────────
    analyzeBtn.addEventListener("click", handleAnalyze);

    // Allow Ctrl+Enter to submit
    reviewInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            handleAnalyze();
        }
    });

    async function handleAnalyze() {
        const text = reviewInput.value.trim();
        if (!text) {
            shakeElement(reviewInput);
            return;
        }

        // Show loader, hide results
        analyzeBtn.disabled = true;
        loader.classList.remove("hidden");
        resultsSection.classList.add("hidden");

        try {
            const response = await fetch("/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ review: text }),
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || "Server error");
            }

            const data = await response.json();
            renderResults(data);
        } catch (error) {
            alert("Analysis failed: " + error.message);
        } finally {
            analyzeBtn.disabled = false;
            loader.classList.add("hidden");
        }
    }

    // ── Render Results ───────────────────────────────────────────────────
    function renderResults(data) {
        const { sentiment, fake_detection, scores, stars } = data;

        // Sentiment card
        const sentMap = {
            Positive: { icon: "😊", cls: "positive" },
            Negative: { icon: "😠", cls: "negative" },
            Neutral:  { icon: "😐", cls: "neutral" },
        };
        const s = sentMap[sentiment.label] || sentMap.Neutral;
        sentimentIcon.textContent = s.icon;
        sentimentLabel.textContent = sentiment.label;
        sentimentLabel.className = `result-label label-${s.cls}`;
        sentimentConf.textContent = `Confidence: ${sentiment.confidence}%`;
        sentimentReason.textContent = sentiment.reason;

        sentimentBar.className = `confidence-bar bar-${s.cls}`;
        sentimentBar.style.width = "0%";
        requestAnimationFrame(() => {
            sentimentBar.style.width = `${sentiment.confidence}%`;
        });

        // Fake detection card
        const fMap = {
            Genuine: { icon: "✅", cls: "genuine" },
            Fake:    { icon: "⚠️", cls: "fake" },
        };
        const f = fMap[fake_detection.label] || fMap.Genuine;
        fakeIcon.textContent = f.icon;
        fakeLabel.textContent = fake_detection.label;
        fakeLabel.className = `result-label label-${f.cls}`;
        fakeConf.textContent = `Confidence: ${fake_detection.confidence}%`;
        fakeReason.textContent = fake_detection.reason;

        fakeBar.className = `confidence-bar bar-${f.cls}`;
        fakeBar.style.width = "0%";
        requestAnimationFrame(() => {
            fakeBar.style.width = `${fake_detection.confidence}%`;
        });

        // Star rating
        renderStars(stars);

        // Chart
        renderChart(scores);

        // Show section
        resultsSection.classList.remove("hidden");
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // ── Star Rating Renderer ─────────────────────────────────────────────
    function renderStars(rating) {
        starDisplay.innerHTML = "";
        const fullStars = Math.floor(rating);
        const hasHalf   = (rating - fullStars) >= 0.3;
        const totalStars = 5;

        for (let i = 0; i < totalStars; i++) {
            const span = document.createElement("span");
            span.classList.add("star", "pop");
            span.style.animationDelay = `${i * 0.1}s`;

            if (i < fullStars) {
                span.textContent = "★";
                span.style.color = "#fbbf24";
            } else if (i === fullStars && hasHalf) {
                span.textContent = "★";
                span.style.color = "#fbbf24";
                span.style.opacity = "0.55";
            } else {
                span.textContent = "★";
                span.style.color = "#374151";
            }
            starDisplay.appendChild(span);
        }

        starValue.textContent = `${rating} / 5.0`;
    }

    // ── Chart Renderer ───────────────────────────────────────────────────
    function renderChart(scores) {
        if (chartInstance) {
            chartInstance.destroy();
        }

        const ctx = chartCanvas.getContext("2d");

        chartInstance = new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Positive", "Negative", "Neutral", "Genuine", "Fake"],
                datasets: [{
                    label: "Confidence (%)",
                    data: [
                        scores.positive,
                        scores.negative,
                        scores.neutral,
                        scores.genuine,
                        scores.fake,
                    ],
                    backgroundColor: [
                        "rgba(52, 211, 153, 0.7)",
                        "rgba(248, 113, 113, 0.7)",
                        "rgba(148, 163, 184, 0.7)",
                        "rgba(34, 211, 238, 0.7)",
                        "rgba(251, 191, 36, 0.7)",
                    ],
                    borderColor: [
                        "#34d399",
                        "#f87171",
                        "#94a3b8",
                        "#22d3ee",
                        "#fbbf24",
                    ],
                    borderWidth: 2,
                    borderRadius: 8,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 1200,
                    easing: "easeOutQuart",
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "rgba(11, 14, 23, 0.9)",
                        titleColor: "#e4e6ef",
                        bodyColor: "#e4e6ef",
                        borderColor: "rgba(255,255,255,0.1)",
                        borderWidth: 1,
                        cornerRadius: 8,
                        padding: 12,
                    },
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { color: "#8b8fa3", font: { family: "Inter" } },
                        grid: { color: "rgba(255,255,255,0.04)" },
                    },
                    x: {
                        ticks: { color: "#8b8fa3", font: { family: "Inter", weight: "600" } },
                        grid: { display: false },
                    },
                },
            },
        });
    }

    // ── Shake animation for empty input ──────────────────────────────────
    function shakeElement(el) {
        el.style.animation = "none";
        el.offsetHeight; // trigger reflow
        el.style.animation = "shake 0.4s ease";
        el.addEventListener("animationend", () => {
            el.style.animation = "";
        }, { once: true });
    }

    // Inject shake keyframes
    const shakeStyle = document.createElement("style");
    shakeStyle.textContent = `
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            20%      { transform: translateX(-8px); }
            40%      { transform: translateX(8px); }
            60%      { transform: translateX(-5px); }
            80%      { transform: translateX(5px); }
        }
    `;
    document.head.appendChild(shakeStyle);
})();
