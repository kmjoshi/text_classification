mkdir -p ~/.streamlit/

echo "\
[server]\n\
enableCORS=false
headless=true\n\
port=$PORT\n\
" > ~/.streamlit/config.toml