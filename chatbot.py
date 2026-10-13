# -*- coding: utf-8 -*-
"""
Olist AI | Smart Assistant
Python + Streamlit + n8n + SQLite

الفكرة:
- التطبيق يقرأ ملف Olist_Master_Dataset.csv الحقيقي.
- يحسب ملخصًا مضغوطًا للبيانات (dataContext).
- يرسل السؤال + الملخص إلى n8n مع كل رسالة.
"""
from dotenv import load_dotenv

load_dotenv()
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

import pandas as pd

df = pd.read_csv("Olist_Master_Dataset.csv")

# Convert delivery dates to datetime
df["order_delivered_customer_date"] = pd.to_datetime(
    df["order_delivered_customer_date"], errors="coerce"
)

df["order_estimated_delivery_date"] = pd.to_datetime(
    df["order_estimated_delivery_date"], errors="coerce"
)

# Select delivered orders with valid dates
delivered = df[
    (df["order_status"] == "delivered")
    & df["order_delivered_customer_date"].notna()
    & df["order_estimated_delivery_date"].notna()
].copy()

# Calculate delay in days
delivered["days_late"] = (
    delivered["order_delivered_customer_date"]
    - delivered["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

# Identify late orders
late = delivered[delivered["days_late"] > 0]

# Calculate metrics safely
total_delivered = len(delivered)
total_late = len(late)

late_percentage = (
    total_late / total_delivered * 100
    if total_delivered > 0 else 0
)

average_days_late = (
    late["days_late"].mean()
    if total_late > 0 else 0
)

print("Total delivered orders:", total_delivered)
print("Late orders:", total_late)
print(f"Late delivery percentage: {late_percentage:.2f}%")
print(f"Average days late among late orders: {average_days_late:.2f}")
# ==================================================
# 1. الإعدادات
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
CSV_PATH = BASE_DIR / "Olist_Master_Dataset.csv"

load_dotenv(ENV_PATH)

# يجب أن يكون هذا أول أمر Streamlit في الملف
st.set_page_config(
    page_title="Olist AI | Smart Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_webhook_url():
    """قراءة رابط n8n من Streamlit Secrets أو ملف .env."""

    try:
        url = st.secrets.get("N8N_WEBHOOK_URL", "")
    except Exception:
        url = ""

    if not url:
        url = os.getenv("N8N_WEBHOOK_URL", "")

    return str(url).strip()


WEBHOOK_URL = get_webhook_url()


# ==================================================
# 2. قراءة بيانات Olist وحساب الملخص
# ==================================================

def to_number(series):
    """تحويل عمود إلى أرقام بأمان (القيم غير الرقمية تصبح NaN)."""
    return pd.to_numeric(series, errors="coerce")


def rnd(value, digits=2):
    """تقريب رقم وتحويله إلى float عادي قابل للإرسال كـ JSON."""
    return round(float(value), digits)


def unique_count(df, column):
    """عدد القيم الفريدة لعمود إن كان موجودًا."""
    if column in df.columns:
        return int(df[column].nunique(dropna=True))
    return None


def numeric_stats(df, column):
    """إحصائيات عمود رقمي (price / freight_value ...)."""
    if column not in df.columns:
        return None

    values = to_number(df[column]).dropna()
    if values.empty:
        return None

    return {
        "non_null_rows": int(values.count()),
        "mean_per_row": rnd(values.mean()),
        "median_per_row": rnd(values.median()),
        "min": rnd(values.min()),
        "max": rnd(values.max()),
        "sum_over_rows": rnd(values.sum()),
        "sum_warning": (
            "المجموع محسوب على صفوف الملف المدموج، وقد يكون مضخمًا "
            "إذا تكررت الطلبات بسبب الدمج (join). "
            "ليس بالضرورة إجمالي مبيعات حقيقي."
        ),
    }


def grouped_counts(df, column, limit=None):
    """
    عدد الطلبات الفريدة لكل قيمة في عمود معين.
    إذا لم يوجد order_id نرجع عدد الصفوف.
    """
    if column not in df.columns:
        return None

    if "order_id" in df.columns:
        counts = (
            df.dropna(subset=[column])
            .groupby(column)["order_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        metric = "unique_orders"
    else:
        counts = df[column].value_counts()
        metric = "rows"

    if limit:
        counts = counts.head(limit)

    return {
        "metric": metric,
        "values": {str(k): int(v) for k, v in counts.items()},
    }


def build_data_summary(df):
    """بناء ملخص مضغوط من البيانات الفعلية (بدون أي أرقام مكتوبة يدويًا)."""

    n_rows, n_cols = df.shape

    summary = {
        "dataset_name": CSV_PATH.name,
        "rows_in_file": int(n_rows),
        "columns_count": int(n_cols),
        "columns": [str(c) for c in df.columns],
        "important_notes": [
            "rows_in_file هو عدد صفوف الملف المدموج وليس عدد الطلبات.",
            "قد يظهر الطلب الواحد في أكثر من صف (منتجات/مدفوعات/تقييمات).",
            "عدد الطلبات الحقيقي هو unique_counts.unique_orders.",
            "هذا ملخص محسوب مسبقًا فقط؛ أي رقم غير موجود هنا غير متاح.",
        ],
    }

    # ---------- القيم الفريدة ----------
    unique_map = {
        "unique_orders": "order_id",
        "unique_customer_ids": "customer_id",
        "unique_customers_by_customer_unique_id": "customer_unique_id",
        "unique_products": "product_id",
        "unique_sellers": "seller_id",
    }
    uniques = {}
    for label, column in unique_map.items():
        value = unique_count(df, column)
        if value is not None:
            uniques[label] = value
    summary["unique_counts"] = uniques

    # ---------- التقييمات ----------
    if "review_score" in df.columns:
        scores = to_number(df["review_score"])
        valid = scores.dropna()

        if not valid.empty:
            review = {
                "average_over_rows": rnd(valid.mean(), 3),
                "rows_with_score": int(valid.count()),
                "min": rnd(valid.min()),
                "max": rnd(valid.max()),
                "score_distribution_over_rows": {
                    str(k): int(v)
                    for k, v in valid.value_counts().sort_index().items()
                },
            }

            if "order_id" in df.columns:
                per_order = scores.groupby(df["order_id"]).mean().dropna()
                if not per_order.empty:
                    review["average_per_order"] = rnd(per_order.mean(), 3)
                    review["orders_with_score"] = int(per_order.shape[0])

            summary["review_score"] = review

    # ---------- السعر والشحن ----------
    for column in ("price", "freight_value", "payment_value"):
        stats = numeric_stats(df, column)
        if stats:
            summary[column] = stats

    # ---------- حالات الطلبات ----------
    status = grouped_counts(df, "order_status")
    if status:
        summary["order_status_counts"] = status

    # ---------- أكثر فئات المنتجات تكرارًا ----------
    category_column = None
    for candidate in (
        "product_category_name_english",
        "product_category_name",
    ):
        if candidate in df.columns:
            category_column = candidate
            break

    if category_column:
        summary["product_category_column_used"] = category_column

        # عدد مرات ظهور كل فئة في صفوف الداتا
        category_rows = (
            df[category_column]
            .dropna()
            .astype(str)
            .str.strip()
        )
        category_rows = category_rows[category_rows != ""]

        summary["top_10_product_categories_by_rows"] = {
            str(category): int(count)
            for category, count in category_rows.value_counts().head(10).items()
        }

        # عدد المنتجات الفريدة في كل فئة
        if "product_id" in df.columns:
            category_products = (
                df[["product_id", category_column]]
                .dropna()
                .drop_duplicates(subset=["product_id"])
            )

            summary["top_10_product_categories_by_unique_products"] = {
                str(category): int(count)
                for category, count in (
                    category_products[category_column]
                    .value_counts()
                    .head(10)
                    .items()
                )
            }

        # عدد الطلبات الفريدة التي ظهرت فيها كل فئة
        if "order_id" in df.columns:
            category_orders = (
                df[["order_id", category_column]]
                .dropna()
                .drop_duplicates()
            )

            summary["top_10_product_categories_by_unique_orders"] = {
                str(category): int(count)
                for category, count in (
                    category_orders.groupby(category_column)["order_id"]
                    .nunique()
                    .sort_values(ascending=False)
                    .head(10)
                    .items()
                )
            }

    # ---------- ولايات العملاء وطرق الدفع ----------
    states = grouped_counts(df, "customer_state", limit=10)
    if states:
        summary["top_10_customer_states"] = states

    payments = grouped_counts(df, "payment_type")
    if payments:
        summary["payment_type_counts"] = payments

    # ---------- التواريخ ----------
    date_column = "order_purchase_timestamp"
    if date_column in df.columns:
        dates = pd.to_datetime(df[date_column], errors="coerce")
        valid_dates = dates.dropna()

        if not valid_dates.empty:
            summary["purchase_date_range"] = {
                "from": valid_dates.min().strftime("%Y-%m-%d"),
                "to": valid_dates.max().strftime("%Y-%m-%d"),
            }

            months = dates.dt.strftime("%Y-%m")
            if "order_id" in df.columns:
                temp = pd.DataFrame(
                    {"order_id": df["order_id"], "month": months}
                ).dropna()
                monthly = temp.groupby("month")["order_id"].nunique()
                metric = "unique_orders"
            else:
                monthly = months.dropna().value_counts().sort_index()
                metric = "rows"

            summary["monthly_orders"] = {
                "metric": metric,
                "values": {
                    str(k): int(v) for k, v in monthly.head(60).items()
                },
            }

    # ---------- القيم المفقودة ----------
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(10)
    if not missing.empty:
        summary["columns_with_most_missing_values"] = {
            str(col): {
                "missing_rows": int(count),
                "missing_percent": rnd(count / n_rows * 100, 1),
            }
            for col, count in missing.items()
        }

    return summary


def get_csv_signature():
    """تاريخ آخر تعديل للملف، حتى يتحدث الكاش عند تغيير الـ CSV."""
    try:
        return CSV_PATH.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner="جاري تحميل بيانات Olist...")
def load_olist_bundle(path_str, mtime):
    """
    قراءة CSV وحساب الملخص مرة واحدة ثم تخزينهما في الكاش.
    ترجع قاموسًا فيه ok و summary أو error، بحيث لا ينهار التطبيق.
    """
    path = Path(path_str)

    if not path.exists():
        return {
            "ok": False,
            "error": f"ملف البيانات غير موجود: {path.name}. "
                     "ضعه في نفس مجلد chatbot.py.",
        }

    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return {"ok": False, "error": "ملف CSV فارغ."}
    except Exception as exc:
        return {"ok": False, "error": f"تعذر قراءة ملف CSV: {exc}"}

    if df.empty:
        return {"ok": False, "error": "ملف CSV لا يحتوي على صفوف."}

    df.columns = [str(c).strip() for c in df.columns]

    try:
        summary = build_data_summary(df)
    except Exception as exc:
        return {"ok": False, "error": f"تعذر حساب ملخص البيانات: {exc}"}

    return {
        "ok": True,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "summary": summary,
    }


DATA = load_olist_bundle(str(CSV_PATH), get_csv_signature())


def detect_language(text):
    """'ar' إذا كان السؤال يحتوي حروفًا عربية، وإلا 'en'."""
    return "ar" if re.search(r"[\u0600-\u06FF]", text or "") else "en"


def build_data_context(question):
    """dataContext الذي يُرسل إلى n8n مع كل سؤال."""

    if DATA.get("ok"):
        context = dict(DATA["summary"])
        context["available"] = True
    else:
        context = {
            "available": False,
            "error": DATA.get("error", "بيانات Olist غير متاحة."),
        }

    context["user_language"] = detect_language(question)
    return context


# ==================================================
# 3. قاعدة بيانات المحادثات
# ==================================================

db_setting = os.getenv("CHATBOT_DB_PATH", "")

if db_setting:
    DB_PATH = Path(db_setting).expanduser()
    if not DB_PATH.is_absolute():
        DB_PATH = BASE_DIR / DB_PATH
else:
    DB_PATH = BASE_DIR / "olist_chat.db"


def get_connection():
    conn = sqlite3.connect(str(DB_PATH), timeout=20)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL
                    CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id, id)
        """)


def create_conversation(title="محادثة جديدة"):
    conversation_id = str(uuid.uuid4())
    timestamp = now_utc()

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO conversations
                (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (conversation_id, title, timestamp, timestamp))

    return conversation_id


def list_conversations():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                c.id,
                c.title,
                c.created_at,
                c.updated_at,
                COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        """).fetchall()

    return [dict(row) for row in rows]


def get_conversation(conversation_id):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE id = ?
        """, (conversation_id,)).fetchone()

    return dict(row) if row else None


def get_messages(conversation_id):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
        """, (conversation_id,)).fetchall()

    return [dict(row) for row in rows]


def save_message(conversation_id, role, content):
    if role not in ("user", "assistant"):
        raise ValueError("نوع الرسالة غير صالح.")

    timestamp = now_utc()

    with get_connection() as conn:
        exists = conn.execute("""
            SELECT id
            FROM conversations
            WHERE id = ?
        """, (conversation_id,)).fetchone()

        if not exists:
            raise ValueError("المحادثة غير موجودة.")

        conn.execute("""
            INSERT INTO messages
                (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (conversation_id, role, str(content), timestamp))

        if role == "user":
            row = conn.execute("""
                SELECT title
                FROM conversations
                WHERE id = ?
            """, (conversation_id,)).fetchone()

            if row and row["title"] == "محادثة جديدة":
                title = str(content).strip().replace("\n", " ")
                if len(title) > 42:
                    title = title[:39] + "..."

                conn.execute("""
                    UPDATE conversations
                    SET title = ?, updated_at = ?
                    WHERE id = ?
                """, (title or "محادثة جديدة",
                      timestamp, conversation_id))
            else:
                conn.execute("""
                    UPDATE conversations
                    SET updated_at = ?
                    WHERE id = ?
                """, (timestamp, conversation_id))
        else:
            conn.execute("""
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
            """, (timestamp, conversation_id))


def delete_conversation(conversation_id):
    with get_connection() as conn:
        conn.execute("""
            DELETE FROM conversations
            WHERE id = ?
        """, (conversation_id,))


init_db()


# ==================================================
# 4. التعامل مع n8n
# ==================================================

def extract_reply(data):
    """استخراج الإجابة من رد n8n (الأفضل: {"answer": "..."})."""

    if isinstance(data, str):
        return data.strip()

    if isinstance(data, list):
        for item in data:
            result = extract_reply(item)
            if result:
                return result
        return ""

    if isinstance(data, dict):
        for key in (
            "answer",
            "reply",
            "output",
            "text",
            "message"
        ):
            value = data.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

            if isinstance(value, (dict, list)):
                result = extract_reply(value)
                if result:
                    return result

    return ""


def ask_n8n(question, conversation_id):
    """إرسال السؤال + ملخص البيانات إلى n8n واستقبال الإجابة."""

    if not WEBHOOK_URL:
        raise ValueError(
            "رابط n8n غير مضبوط. "
            "تأكد من ملف .env ووجود N8N_WEBHOOK_URL."
        )

    payload = {
        "question": question,
        "chatInput": question,
        "sessionId": conversation_id,
        "dataContext": build_data_context(question),
    }

    try:
        response = requests.post(
            WEBHOOK_URL,
            json=payload,
            timeout=90
        )
        response.raise_for_status()

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "الاتصال بـ n8n استغرق وقتًا طويلًا."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "؟"
        raise RuntimeError(
            f"n8n أعاد خطأ HTTP {status}. "
            "راجع رابط Production وحالة الـ Workflow."
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"تعذر الاتصال بـ n8n: {exc}"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "رد n8n ليس بصيغة JSON صحيحة."
        ) from exc

    answer = extract_reply(data)

    if not answer:
        raise RuntimeError(
            "وصل رد من n8n، لكن لم يتم العثور على الإجابة. "
            "راجع Respond to Webhook."
        )

    return answer


# ==================================================
# 5. تصميم الواجهة
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #25204a 0%, transparent 35%),
        radial-gradient(ellipse at 95% 20%, #102d48 0%, transparent 30%),
        #0b1020;
    color: #f8fafc;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1100px;
    padding-top: 1.5rem;
    padding-bottom: 5rem;
}

[data-testid="stSidebar"] {
    background: #0c1223;
    border-right: 1px solid #26304a;
}

/* عناوين ونصوص عامة واضحة */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #f8fafc !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #cbd5e1 !important;
}

.brand, .topbar {
    padding: 18px;
    border: 1px solid #34305d;
    border-radius: 18px;
    background: linear-gradient(135deg, #211c42, #10253c);
    margin-bottom: 18px;
}

.brand-title {
    font-size: 27px;
    font-weight: 800;
    color: #c4b5fd;
}

.brand-sub, .top-sub {
    color: #a5b4fc;
    font-size: 12px;
}

.top-title {
    font-size: 24px;
    font-weight: 800;
    color: #f8fafc;
}

.welcome-box, .welcome-box h2, .welcome-box p {
    color: #f8fafc !important;
}

/* ---------- رسائل الشات ---------- */
[data-testid="stChatMessage"] {
    background: rgba(20, 29, 49, .9);
    border: 1px solid #2b3853;
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 10px;
}

[data-testid="stChatMessage"],
[data-testid="stChatMessageContent"],
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span,
[data-testid="stChatMessageContent"] strong,
[data-testid="stChatMessageContent"] em,
[data-testid="stChatMessageContent"] td,
[data-testid="stChatMessageContent"] th,
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3,
[data-testid="stChatMessageContent"] h4,
[data-testid="stChatMessageContent"] h5,
[data-testid="stChatMessageContent"] h6,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong {
    color: #f8fafc !important;
}

[data-testid="stChatMessageContent"] {
    line-height: 1.9;
}

[data-testid="stChatMessageContent"] a {
    color: #a5b4fc !important;
}

[data-testid="stChatMessageContent"] code {
    color: #e9d5ff !important;
    background: #1e1b3a !important;
    border-radius: 6px;
    padding: 1px 6px;
}

[data-testid="stChatMessageContent"] pre {
    background: #0f172a !important;
    border: 1px solid #2b3853;
    border-radius: 10px;
}

[data-testid="stChatMessageContent"] table,
[data-testid="stChatMessageContent"] th,
[data-testid="stChatMessageContent"] td {
    border: 1px solid #334155 !important;
}

/* ---------- خانة إدخال الشات ---------- */
[data-testid="stBottom"] > div {
    background: transparent !important;
}

[data-testid="stChatInput"] {
    background: #131d32 !important;
    border: 1px solid #475569;
    border-radius: 16px;
}

[data-testid="stChatInput"] div[data-baseweb="textarea"],
[data-testid="stChatInput"] div[data-baseweb="base-input"] {
    background: #131d32 !important;
}

[data-testid="stChatInput"] textarea {
    background: #131d32 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #cbd5e1 !important;
    -webkit-text-fill-color: #cbd5e1 !important;
    opacity: 1 !important;
}

.stButton > button {
    border-radius: 12px;
    min-height: 40px;
    border: 1px solid #384562;
    background: linear-gradient(135deg, #18243c, #211c42);
    color: #e2e8f0;
    font-family: 'Cairo', sans-serif;
}

.stButton > button:hover {
    border-color: #8b5cf6;
    color: white;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 11px;
    padding: 20px 0;
}

hr {
    border-color: #28344c;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 6. تهيئة حالة المحادثة
# ==================================================

if "conversation_id" not in st.session_state:
    conversations = list_conversations()

    if conversations:
        st.session_state.conversation_id = conversations[0]["id"]
    else:
        st.session_state.conversation_id = create_conversation()

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "delete_confirmation" not in st.session_state:
    st.session_state.delete_confirmation = None

if not get_conversation(st.session_state.conversation_id):
    st.session_state.conversation_id = create_conversation()


# ==================================================
# 7. القائمة الجانبية
# ==================================================

with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div style="font-size:32px">✦</div>
        <div class="brand-title">Olist AI</div>
        <div class="brand-sub">YOUR INTELLIGENT BUSINESS ASSISTANT</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("＋ محادثة جديدة", use_container_width=True):
        st.session_state.conversation_id = create_conversation()
        st.session_state.pending_prompt = None
        st.session_state.delete_confirmation = None
        st.rerun()

    st.divider()
    st.markdown("### 🕘 المحادثات السابقة")

    for conversation in list_conversations():
        cid = conversation["id"]
        title = conversation["title"]
        count = conversation["message_count"]

        col_open, col_delete = st.columns([5, 1])

        with col_open:
            label = f"💬 {title}"
            if cid == st.session_state.conversation_id:
                label = f"🔹 {title}"

            if st.button(
                label,
                key=f"open_{cid}",
                use_container_width=True,
                help=f"عدد الرسائل: {count}"
            ):
                st.session_state.conversation_id = cid
                st.session_state.pending_prompt = None
                st.session_state.delete_confirmation = None
                st.rerun()

        with col_delete:
            if st.button("🗑️", key=f"delete_{cid}"):
                st.session_state.delete_confirmation = cid
                st.rerun()

        if st.session_state.delete_confirmation == cid:
            st.warning(f"هل تريد حذف «{title}»؟")
            yes_col, no_col = st.columns(2)

            with yes_col:
                if st.button("نعم، احذف", key=f"confirm_{cid}"):
                    delete_conversation(cid)

                    if st.session_state.conversation_id == cid:
                        st.session_state.conversation_id = create_conversation()

                    st.session_state.delete_confirmation = None
                    st.rerun()

            with no_col:
                if st.button("إلغاء", key=f"cancel_{cid}"):
                    st.session_state.delete_confirmation = None
                    st.rerun()

    st.divider()

    # ---------- حالة n8n ----------
    if WEBHOOK_URL:
        st.success("رابط n8n مضبوط")
    else:
        st.warning("رابط n8n غير مضبوط")

    # ---------- حالة بيانات Olist ----------
    st.markdown("### 📊 بيانات Olist")

    if DATA.get("ok"):
        summary = DATA["summary"]
        uniques = summary.get("unique_counts", {})

        st.success(
            f"تم تحميل البيانات: {DATA['rows']:,} صف × {DATA['cols']} عمود"
        )

        if "unique_orders" in uniques:
            st.caption(f"طلبات فريدة: {uniques['unique_orders']:,}")

        if "unique_customers_by_customer_unique_id" in uniques:
            st.caption(
                "عملاء فريدون: "
                f"{uniques['unique_customers_by_customer_unique_id']:,}"
            )
        elif "unique_customer_ids" in uniques:
            st.caption(
                f"عملاء فريدون (customer_id): "
                f"{uniques['unique_customer_ids']:,}"
            )

        if "unique_products" in uniques:
            st.caption(f"منتجات فريدة: {uniques['unique_products']:,}")

        if "unique_sellers" in uniques:
            st.caption(f"بائعون فريدون: {uniques['unique_sellers']:,}")

        review = summary.get("review_score")
        if review:
            value = review.get("average_per_order",
                               review.get("average_over_rows"))
            st.caption(f"متوسط التقييم: {value}")

        date_range = summary.get("purchase_date_range")
        if date_range:
            st.caption(
                f"فترة الشراء: {date_range['from']} → {date_range['to']}"
            )

        st.caption(
            "ملاحظة: عدد الصفوف أكبر من عدد الطلبات بسبب دمج الجداول."
        )

        context_size = len(json.dumps(summary, ensure_ascii=False))
        st.caption(f"حجم الملخص المُرسل لكل سؤال: ~{context_size:,} حرف")
    else:
        st.error(DATA.get("error", "تعذر تحميل بيانات Olist."))
        st.caption("سيعمل الشات، لكن بدون سياق بيانات.")

    st.caption("Olist AI • Python + Streamlit + n8n + SQLite")


# ==================================================
# 8. العنوان والأسئلة المقترحة
# ==================================================

st.markdown("""
<div class="topbar">
    <div class="top-title">🤖 Olist AI Assistant</div>
    <div class="top-sub">مساعدك الذكي لتحليل الأعمال والبيانات</div>
</div>
""", unsafe_allow_html=True)

current_id = st.session_state.conversation_id
messages = get_messages(current_id)

if not messages:
    st.markdown("""
    <div class="welcome-box" style="text-align:center; padding:25px 8px;">
        <div style="font-size:42px;">✦</div>
        <h2>أهلًا بك في Olist AI</h2>
        <p>محادثاتك محفوظة تلقائيًا. اكتب سؤالك أو اختر سؤالًا مقترحًا.</p>
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        "كم عدد الطلبات الفريدة في البيانات؟",
        "ما متوسط تقييمات العملاء؟",
        "ما أكثر فئات المنتجات تكرارًا؟",
        "ما توزيع حالات الطلبات؟"
    ]

    col1, col2 = st.columns(2)

    for i, prompt in enumerate(suggestions):
        with (col1 if i % 2 == 0 else col2):
            if st.button(
                prompt,
                key=f"suggestion_{i}",
                use_container_width=True
            ):
                st.session_state.pending_prompt = prompt
                st.rerun()


# ==================================================
# 9. عرض الرسائل السابقة
# ==================================================

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==================================================
# 10. استقبال السؤال وإرسال الطلب إلى n8n
# ==================================================

typed_prompt = st.chat_input("✨ اكتب سؤالك هنا...")

if st.session_state.pending_prompt is not None:
    question = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
else:
    question = typed_prompt


if question and question.strip():
    question = question.strip()
    current_id = st.session_state.conversation_id

    save_message(current_id, "user", question)

    with st.chat_message("assistant"):
        with st.spinner("✦ جاري تجهيز الإجابة..."):
            try:
                answer = ask_n8n(question, current_id)
                st.markdown(answer)

            except (ValueError, RuntimeError) as error:
                answer = f"⚠️ {error}"
                st.error(answer)

            except Exception as error:
                answer = "حدث خطأ غير متوقع أثناء معالجة السؤال."
                st.error(answer)
                print(f"Unexpected chatbot error: {error}")

    save_message(current_id, "assistant", answer)
    st.rerun()


# ==================================================
# 11. تذييل الصفحة
# ==================================================

st.markdown("""
<div class="footer">
    Olist AI Assistant © 2026<br>
    Python • Streamlit • n8n • SQLite
</div>
""", unsafe_allow_html=True)