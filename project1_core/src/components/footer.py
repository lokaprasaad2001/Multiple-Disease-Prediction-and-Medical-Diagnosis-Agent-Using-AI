import streamlit as st
from project1_core.src.config.app_config import PRIMARY_COLOR, SECONDARY_COLOR
import requests
import time

def get_github_stars():
    try:
        response = requests.get("https://api.github.com/repos/harshhh28/hia")
        if response.status_code == 200:
            return response.json()["stargazers_count"]
        return None
    except:
        return None

def show_footer(in_sidebar=False):
    # Cache the stars count for 1 hour
    @st.cache_data(ttl=36000)
    def get_cached_stars():
        return get_github_stars()
    
    stars_count = 0
    
    base_styles = f"""
        text-align: center;
        padding: 0.75rem;
        background: linear-gradient(to right, 
            rgba(25, 118, 210, 0.03), 
            rgba(100, 181, 246, 0.05), 
            rgba(25, 118, 210, 0.03)
        );
        border-top: 1px solid rgba(100, 181, 246, 0.15);
        margin-top: {'0' if in_sidebar else '2rem'};
        {'width: 100%' if not in_sidebar else ''};
        box-shadow: 0 -2px 10px rgba(100, 181, 246, 0.05);
    """
    
    st.markdown(
        f"""
        <div style='{base_styles}'>
            <p style='
                font-family: "Source Sans Pro", sans-serif;
                color: #64B5F6;
                font-size: 0.75rem;
                letter-spacing: 0.02em;
                margin: 0;
                opacity: 0.95;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 8px;
            '>
                <span style="
                    display: flex;
                    align-items: center;
                    gap: 4px;
                    padding: 2px 8px;
                    border-radius: 4px;
                    background: rgba(100, 181, 246, 0.05);
                    transition: all 0.2s ease;
                ">
                    <a href='#' 
                       target='_blank' 
                       style='
                           color: #64B5F6;
                           text-decoration: none;
                           font-weight: 500;
                           transition: all 0.2s ease;
                           display: inline-flex;
                           align-items: center;
                           gap: 4px;
                       '
                       onmouseover="this.style.color='{PRIMARY_COLOR}'; this.style.textDecoration='underline'"
                       onmouseout="this.style.color='#1976D2'; this.style.textDecoration='none'">
                        <span style="color: #64B5F6;">Contribute to</span>
                        WORLD
                        {f'<span style="display: inline-flex; align-items: center; gap: 4px; margin-left: 4px; color: #64B5F6;"><svg height="12" width="12" viewBox="0 0 16 16" fill="#64B5F6"><path d="M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25z"></path></svg>{stars_count}</span>' if stars_count is not None else ''}
                    </a>
                </span>
                <span style="
                    color: #1976D2;
                    display: flex;
                    align-items: center;
                    gap: 4px;
                    transition: all 0.2s ease;
                ">
                    <a href='#' 
                       target='_blank' 
                       style='
                           color: #1976D2;
                           text-decoration: none;
                           font-weight: 500;
                           transition: all 0.2s ease;
                           display: inline-flex;
                           align-items: center;
                           gap: 4px;
                       '
                       onmouseover="this.style.color='{PRIMARY_COLOR}'; this.style.textDecoration='underline'"
                       onmouseout="this.style.color='#1976D2'; this.style.textDecoration='none'">
                       AI @ SERVICE
                    </a>
                </span>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
