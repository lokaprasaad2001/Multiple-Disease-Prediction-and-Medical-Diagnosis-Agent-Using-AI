# File: project2/project1_core/src/main.py

import streamlit as st

# --- IMPORTANT CHANGES TO IMPORTS BELOW ---
# All these imports now need to be absolute relative to the 'project1_core' package root
from project1_core.src.auth.session_manager import SessionManager
from project1_core.src.components.auth_pages import show_login_page
from project1_core.src.components.sidebar import show_sidebar
from project1_core.src.components.analysis_form import show_analysis_form
from project1_core.src.components.footer import show_footer
from project1_core.src.config.app_config import APP_NAME, APP_TAGLINE, APP_DESCRIPTION, APP_ICON

# Must be the first Streamlit command (keep it for this page's specific config)
st.set_page_config(
    page_title="HIA - Health Insights Agent",
    page_icon="🩺",
    layout="wide"
)

# Initialize session state
#SessionManager.init_session()

# Hide all Streamlit form-related elements
st.markdown("""
    <style>
        /* Hide form submission helper text */
        div[data-testid="InputInstructions"] > span:nth-child(1) {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

def show_welcome_screen():
    st.markdown(
        f"""
        <div style='text-align: center; padding: 50px;'>
            <h1>{APP_ICON} {APP_NAME}</h1>
            <h3>{APP_DESCRIPTION}</h3>
            <p style='font-size: 1.2em; color: #666;'>{APP_TAGLINE}</p>
            <p>Start by creating a new analysis session</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        if st.button("➕ Create New Analysis Session", use_container_width=True, type="primary"):
            success, session = SessionManager.create_chat_session()
            if success:
                st.session_state.current_session = session
                st.rerun()
            else:
                st.error("Failed to create session")

def show_chat_history():
    if 'auth_service' in st.session_state and 'current_session' in st.session_state:
        success, messages = st.session_state.auth_service.get_session_messages(
            st.session_state.current_session['id']
        )
        
        if success:
            for msg in messages:
                if msg['role'] == 'user':
                    st.info(msg['content'])
                else:
                    st.success(msg['content'])
    else:
        st.warning("Session or authentication service not initialized. Please log in or create a session.")

def show_user_greeting():
    if st.session_state.user:
        display_name = st.session_state.user.get('name') or st.session_state.user.get('email', '')
        st.markdown(f"""
            <div style='text-align: right; padding: 1rem; color: #64B5F6; font-size: 1.1em;'>
                👋 Hi, {display_name}
            </div>
        """, unsafe_allow_html=True)

# Encapsulate the main logic into a function
def run_hia_dashboard():
    SessionManager.init_session() # Already called globally above

    if not SessionManager.is_authenticated():
        show_login_page()
        show_footer()
        return

    show_user_greeting()
    show_sidebar()

    if st.session_state.get('current_session'):
        st.title(f"📊 {st.session_state.current_session['title']}")
        show_chat_history()
        show_analysis_form()
    else:
        show_welcome_screen()

# Remove the __main__ block
# if __name__ == "__main__":
#     main()