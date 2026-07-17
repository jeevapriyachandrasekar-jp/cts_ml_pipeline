const data = {
  stores: [
    ['Store #018 — West Michael', 'Kiosk', 65.55],
    ['Store #016 — Port Jesseville', 'Drive-Thru', 65.39],
    ['Store #023 — East Lydiamouth', 'Dine-In', 64.28],
    ['Store #004 — East Jessetown', 'Dine-In', 62.40]
  ],
  categories: [
    ['Chicken', 2500], ['Burgers', 1225], ['Breakfast', 1160], ['Sides', 1040], ['Desserts', 586], ['Beverages', 473]
  ],
  products: ['Spicy Chicken Sandwich Combo (5.93 units)', 'Crispy Chicken Sandwich Combo (5.92 units)', 'Hotcakes Combo (5.73 units)', 'Latte Combo (5.72 units)', 'Large Fries Combo (5.67 units)']
};

document.querySelectorAll('.mini-bars').forEach(el => {
  el.dataset.bars.split(',').forEach(height => { const bar = document.createElement('span'); bar.style.height = `${height * 2}px`; el.append(bar); });
});

const storeRows = document.getElementById('storeRows');
if (storeRows) storeRows.innerHTML = data.stores.map((s, i) => `<div class="tr"><div><div class="store-name">${s[0]}</div><div class="store-city">${i === 0 ? 'Leeville, PA' : i === 1 ? 'New David, IL' : i === 2 ? 'Taylorburgh, MI' : 'North Jessicaland, IL'}</div></div><span class="type-tag">${s[1]}</span><span class="forecast">$${s[2].toFixed(2)}</span></div>`).join('');
const maxGrowth = 2500;
const categoryRows = document.getElementById('categoryRows');
if (categoryRows) categoryRows.innerHTML = data.categories.map(c => `<div class="category-row"><b>${c[0]}</b><div class="progress"><span style="width:${Math.max(8, c[1] / maxGrowth * 100)}%"></span></div><span class="growth">+${c[1].toLocaleString()}%</span></div>`).join('');

const values = [36,40,37,44,42,47,48,45,53,50,56,55,61,57,64,60,70,66,75,77,73,81,78,84,82,91,88,95];
const future = [88,94,92,100,97,106,104];
const denominator = values.length + future.length - 1;
const points = (a, offset = 0) => a.map((v,i)=>`${((i + offset) / denominator * 100).toFixed(2)},${(100-v).toFixed(2)}`).join(' ');
const historical = points(values), forecast = points([...values.slice(-1), ...future], values.length - 1);
const lineChart = document.getElementById('lineChart');
if (lineChart) lineChart.innerHTML = `<svg viewBox="0 0 100 100" preserveAspectRatio="none"><defs><linearGradient id="fill" x1="0" x2="0" y1="0" y2="1"><stop stop-color="#49e7d0" stop-opacity=".25"/><stop offset="1" stop-color="#49e7d0" stop-opacity="0"/></linearGradient></defs><polygon points="0,100 ${historical} ${(values.length-1)/(values.length+future.length-1)*100},100" fill="url(#fill)"/><polyline points="${historical}" fill="none" stroke="#49e7d0" stroke-width="1.3" vector-effect="non-scaling-stroke"/><polyline points="${forecast}" fill="none" stroke="#779cff" stroke-width="1.3" stroke-dasharray="5 5" vector-effect="non-scaling-stroke"/></svg>`;
const productPredictions = document.getElementById('productPredictions');
if (productPredictions) productPredictions.innerHTML = data.products.map((product, index) => `<div class="product-card"><span>0${index + 1}</span><b>${product.split(' (')[0]}</b><strong>${product.match(/\((.*?)\)/)[1]}</strong><small>${index < 2 ? 'Chicken' : index === 2 ? 'Breakfast' : index === 3 ? 'McCafe' : 'Sides'}</small></div>`).join('');

const assistant = document.getElementById('assistant'), messages = document.getElementById('messages'), input = document.getElementById('assistantInput');
const toggleAssistant = open => { if (!assistant) return; assistant.classList.toggle('open', open); if(open) input.focus(); };
if (assistant) {
document.getElementById('openAssistant').onclick = () => toggleAssistant(true);
document.getElementById('closeAssistant').onclick = () => toggleAssistant(false);
document.addEventListener('keydown', e => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); toggleAssistant(true); } if(e.key === 'Escape') toggleAssistant(false); });
function answer(question) {
  const q = question.toLowerCase();
  if (/attention|today|risk/.test(q)) return 'Focus on the 10 low-performing stores. They sit in the bottom revenue quartile. For growth, breakfast is the clearest opportunity: unit demand rose 1,160% over the prior 30 days.';
  if (/stock|product|item|demand/.test(q)) return `Prioritise these products for tomorrow: ${data.products.slice(0,3).join(', ')}. The product-demand model has an R² of 0.7079, so treat this as a useful planning signal—not an exact order quantity.`;
  if (/breakfast/.test(q)) return 'Breakfast is growing quickly, led by Hotcakes Combo at 5.7 predicted units tomorrow. Test additional availability in early-day windows, then monitor waste and sell-through before expanding broadly.';
  if (/store|revenue|leader/.test(q)) return 'Store #018 — West Michael is forecast to lead tomorrow at $65.55, followed closely by Store #016 — Port Jesseville at $65.39. Both are good benchmarks for operational comparison.';
  if (/customer|segment|loyal/.test(q)) return 'There are 219 high-frequency customers. They average $87.64 in spending and 4.67 purchases, versus $28.73 and 2.04 for the occasional segment. Loyalty offers and personalised reactivation are the highest-leverage actions.';
  if (/model|performance|r.?2|accur/.test(q)) return 'Store revenue is the strongest forecast (R² 0.7335), followed by product demand (0.7079) and weekly sales (0.6450). Daily sales is weaker (0.2509), so use daily totals as directional guidance and rely more on store/product forecasts for decisions.';
  if (/categor|trend|grow/.test(q)) return 'Every tracked category is growing. Chicken leads at +2,500% unit growth, followed by Burgers (+1,225%), Breakfast (+1,160%), and Sides (+1,040%). These percentages start from small prior-month volumes, so confirm them against absolute units.';
  return 'I can help with sales trends, store forecasts, product stocking, customer segments, or model performance. Try: “What needs attention today?”';
}
function ask(question) { if (!question.trim()) return; messages.insertAdjacentHTML('beforeend', `<div class="message user"></div>`); messages.lastElementChild.textContent = question; setTimeout(() => { messages.insertAdjacentHTML('beforeend', `<div class="message ai"></div>`); messages.lastElementChild.textContent = answer(question); messages.scrollTop = messages.scrollHeight; }, 260); messages.scrollTop = messages.scrollHeight; }
document.getElementById('assistantForm').onsubmit = e => { e.preventDefault(); ask(input.value); input.value = ''; };
document.querySelectorAll('[data-prompt]').forEach(button => button.onclick = () => { toggleAssistant(true); ask(button.dataset.prompt); });
}
