import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

BACKEND_URL = "http://127.0.0.1:8000/chat"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Search Assistant",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .product-card {
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
        height: 100%;
        background-color: #ffffff;
    }

    .product-title {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .product-manufacturer {
        font-size: 14px;
        color: #555;
        margin-bottom: 5px;
    }

    .product-sku {
        font-size: 13px;
        color: #777;
        margin-bottom: 10px;
    }

    .product-description {
        font-size: 14px;
        color: #444;
        line-height: 1.5;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title("🔎 AI Search Assistant")

st.write(
    "Search products or ask general questions."
)


# ============================================================
# QUERY INPUT
# ============================================================

query = st.chat_input(
    "Ask something..."
)


# ============================================================
# SEND QUERY TO BACKEND
# ============================================================

if query:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.write(query)

    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        try:

            # =================================================
            # BACKEND REQUEST
            # =================================================

            with st.spinner("Thinking..."):

                response = requests.post(
                    BACKEND_URL,
                    json={
                        "query": query
                    },
                    timeout=120
                )

            # =================================================
            # STATUS CHECK
            # =================================================

            if response.status_code != 200:

                st.error(
                    f"Backend error: {response.status_code}"
                )

                st.code(
                    response.text
                )

                st.stop()

            # =================================================
            # JSON RESPONSE
            # =================================================

            data = response.json()

            answer = data.get(
                "answer",
                ""
            )

            products = data.get(
                "products",
                []
            )

            # =================================================
            # DISPLAY ANSWER
            # =================================================

            if answer:

                st.write(answer)

            # =================================================
            # DISPLAY PRODUCTS
            # =================================================

            if products:

                st.subheader(
                    f"Products ({len(products)})"
                )

                # Create 3 columns
                columns = st.columns(3)

                for index, product in enumerate(products):

                    column = columns[index % 3]

                    with column:

                        # -------------------------------------
                        # PRODUCT IMAGE
                        # -------------------------------------

                        image_url = product.get(
                            "image_url"
                        )

                        if image_url:

                            # Handle Markdown image URLs
                            # if backend ever returns:
                            # [url](url)

                            if image_url.startswith("["):

                                try:

                                    image_url = (
                                        image_url
                                        .split("](")[1]
                                        .rstrip(")")
                                    )

                                except Exception:
                                    pass

                            st.image(
                                image_url,
                                use_container_width=True
                            )

                        else:

                            st.info(
                                "No image available"
                            )

                        # -------------------------------------
                        # PRODUCT NAME
                        # -------------------------------------

                        name = product.get(
                            "name",
                            "Unknown Product"
                        )

                        st.markdown(
                            f"### {name}"
                        )

                        # -------------------------------------
                        # MANUFACTURER
                        # -------------------------------------

                        manufacturer = product.get(
                            "manufacturer"
                        )

                        if manufacturer:

                            st.write(
                                f"**Manufacturer:** "
                                f"{manufacturer}"
                            )

                        # -------------------------------------
                        # SKU
                        # -------------------------------------

                        sku = product.get(
                            "sku"
                        )

                        if sku:

                            st.write(
                                f"**SKU:** {sku}"
                            )

                        # -------------------------------------
                        # DESCRIPTION
                        # -------------------------------------

                        description = product.get(
                            "description"
                        )

                        if description:

                            st.write(
                                description
                            )

                        # -------------------------------------
                        # PRODUCT ID
                        # -------------------------------------

                        product_id = product.get(
                            "id"
                        )

                        if product_id:

                            st.caption(
                                f"Product ID: {product_id}"
                            )

            # =================================================
            # NO PRODUCTS
            # =================================================

            elif "product" in query.lower():

                st.info(
                    "No matching products were found."
                )

        # ====================================================
        # REQUEST ERROR
        # ====================================================

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the backend. "
                "Make sure FastAPI is running."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The request timed out. "
                "Please try again."
            )

        except Exception as e:

            st.error(
                f"Something went wrong: {str(e)}"
            )