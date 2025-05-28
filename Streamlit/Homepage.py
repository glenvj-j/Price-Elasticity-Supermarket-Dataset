import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import request

st.set_page_config(
    page_title="Price Elasticity Calculator",
    layout="wide"
)

st.sidebar.markdown(
    """
    <h1>Price Elasticity Calculator</h1>
    Discover the ideal selling price to achieve the highest profit.
    <hr>
    """,
    unsafe_allow_html=True
)

# --------------- Modeling ---------------

def model(df_grouped):
    from pygam import ExpectileGAM, s, f
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import LabelEncoder
    from itertools import product
    import numpy as np

    if not df_grouped.empty:
        # Prepare your data
        X_price = df_grouped[['Unit Selling Price (RMB/kg)']].values
        X_event = df_grouped['Event'].values

        # Encode categorical events - do this BEFORE splitting data
        event_encoder = LabelEncoder()
        X_event_encoded = event_encoder.fit_transform(X_event).reshape(-1, 1)

        # Combine features
        X = np.column_stack([X_price, X_event_encoded])
        y_raw = df_grouped['Quantity Sold (kilo)'].values
        y = np.log1p(y_raw)  # Log-transform target

        # Define parameter grid for grid search
        param_grid = {
            'lam': [0.1, 1, 100],
            'n_splines': [10, 20, 30],
            'expectile': [0.25, 0.5, 0.75]
        }

        # Custom scoring function for expectile regression
        def expectile_score(y_true, y_pred, expectile):
            residuals = y_true - y_pred
            return np.mean(np.where(residuals >= 0, 
                                 expectile * residuals, 
                                 (expectile - 1) * residuals))

        # K-Fold cross-validation with consistent encoding
        def grid_search_cv(X, y, param_grid, n_splits=5):
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            best_params = {}
            
            for expectile in param_grid['expectile']:
                print(f"\nGrid searching for expectile: {expectile}")
                best_score = np.inf
                current_best_params = None
                
                for lam, n_splines in product(param_grid['lam'], param_grid['n_splines']):
                    scores = []
                    
                    for train_idx, val_idx in cv.split(X):
                        X_train, X_val = X[train_idx], X[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]
                        
                        # Create GAM with both spline and factor terms
                        gam = ExpectileGAM(
                            s(0, n_splines=n_splines) + f(1),  # s(0) for price, f(1) for event
                            lam=lam,
                            expectile=expectile
                        )
                        
                        # Ensure validation set contains all categories
                        unique_train = np.unique(X_train[:, 1])
                        unique_val = np.unique(X_val[:, 1])
                        
                        # If validation set has categories not in training, skip this split
                        if not set(unique_val).issubset(set(unique_train)):
                            print(f"Skipping split - validation contains unseen categories")
                            continue
                            
                        gam.fit(X_train, y_train)
                        
                        y_pred = gam.predict(X_val)
                        score = expectile_score(y_val, y_pred, expectile)
                        scores.append(score)
                    
                    if not scores:  # If all splits were skipped
                        continue
                        
                    mean_score = np.mean(scores)
                    print(f"lam: {lam}, n_splines: {n_splines} - Score: {mean_score:.4f}")
                    
                    if mean_score < best_score:
                        best_score = mean_score
                        current_best_params = {'lam': lam, 'n_splines': n_splines}
                
                if current_best_params:  # Only store if we found valid params
                    best_params[expectile] = current_best_params
                    print(f"Best params for expectile {expectile}: {current_best_params}")
            
            return best_params

        # Run grid search
        best_params = grid_search_cv(X, y, param_grid)

        if not best_params:  # Handle case where no valid models were found
            st.error("Could not find valid model parameters. Check your data and categories.")
            return None

        # Fit final models with best parameters
        final_gam_results = {}
        for expectile in best_params.keys():
            params = best_params[expectile]
            gam = ExpectileGAM(
                s(0, n_splines=params['n_splines']) + f(1),  # Price + Event
                lam=params['lam'],
                expectile=expectile
            )
            gam.fit(X, y)
            
            # Predict and inverse transform
            y_pred_log = gam.predict(X)
            y_pred = np.expm1(y_pred_log)
            final_gam_results[f'pred_{expectile}'] = y_pred

        # Create final dataframe
        prediction_gam_df = pd.concat(
            [df_grouped, pd.DataFrame(final_gam_results, index=df_grouped.index)],
            axis=1
        )[[
            'Unit Selling Price (RMB/kg)', 'Item Name', 'Category Name', 'Event', 'Wholesale Price (RMB/kg)',
            'pred_0.25', 'pred_0.5', 'pred_0.75'
        ]]

        # Enforce monotonicity
        def enforce_monotonicity(row):
            preds = sorted([row['pred_0.25'], row['pred_0.5'], row['pred_0.75']])
            row['pred_0.25'], row['pred_0.5'], row['pred_0.75'] = preds
            return row

        prediction_gam_df = prediction_gam_df.apply(enforce_monotonicity, axis=1)

        st.sidebar.markdown('Best Parameters found:')
        st.sidebar.dataframe(pd.DataFrame(best_params).T)

        return prediction_gam_df

# --------------- Reg Preview ---------------

def reg_preview(df_clean) : 
    # ---------- Preview regression

    import plotly.express as px
    import statsmodels.api as sm

    # Prepare the data
    df_clean = df_grouped.dropna(subset=['Unit Selling Price (RMB/kg)', 'Quantity Sold (kilo)'])
    X = df_clean['Unit Selling Price (RMB/kg)']
    y = df_clean['Quantity Sold (kilo)']

    # Fit regression line
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()
    df_clean['Regression'] = model.predict(X_sm)

    # Create Plotly figure
    fig = px.scatter(
        df_clean,
        x='Unit Selling Price (RMB/kg)',
        y='Quantity Sold (kilo)',
        title="Regression Preview (All Event)",
        labels={'Quantity Sold (kilo)': 'Quantity Sold', 'Unit Selling Price (RMB/kg)': 'Unit Price'},
        width=800, height=300
    )

    # Add regression line
    fig.add_scatter(
        x=df_clean['Unit Selling Price (RMB/kg)'],
        y=df_clean['Regression'],
        mode='lines',
        name='Regression Line',
        line=dict(color='red')
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# --------------- Show Prediction Chart ---------------

def show_prediction(df_grouped):
    import plotly.graph_objects as go
    import plotly.express as px

    if not df_grouped.empty and selected_event:
        df_view = prediction_gam_df[prediction_gam_df['Event'].isin(selected_event)]
        st.sidebar.markdown(f'Total Data After Filter: {df_view.shape[0]}, Loss Rate : {loss_rate_selected.values[0]}')

        # Calculate predicted sales value = predicted quantity × price
        df_view["y_median"] = df_view["pred_0.5"] * (df_view["Unit Selling Price (RMB/kg)"]-(df_view["Wholesale Price (RMB/kg)"]/(1-loss_rate_selected.values[0])))
        df_view["y_upper"] = df_view["pred_0.75"] * (df_view["Unit Selling Price (RMB/kg)"]-(df_view["Wholesale Price (RMB/kg)"]/(1-loss_rate_selected.values[0])))
        df_view["y_lower"] = df_view["pred_0.25"] * (df_view["Unit Selling Price (RMB/kg)"]-(df_view["Wholesale Price (RMB/kg)"]/(1-loss_rate_selected.values[0])))

        df_view.sort_values(by="Unit Selling Price (RMB/kg)", inplace=True)

        # Convert colors to rgba
        def hex_rgba(hex, transparency):
            col_hex = hex.lstrip('#')
            col_rgb = list(int(col_hex[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({col_rgb[0]},{col_rgb[1]},{col_rgb[2]},{transparency})'

        colors = px.colors.qualitative.Plotly
        fill_color = hex_rgba(colors[0], 0.2)
        line_color = hex_rgba(colors[0], 1.0)

        # Create figure
        fig = go.Figure()

        x = df_view["Unit Selling Price (RMB/kg)"]
        y_median = df_view["y_median"]
        y_upper = df_view["y_upper"]
        y_lower = df_view["y_lower"]

        # Add shaded quantile area
        fig.add_trace(go.Scatter(
            x=pd.concat([x, x[::-1]]),
            y=pd.concat([y_upper, y_lower[::-1]]),
            fill='toself',
            fillcolor=fill_color,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='25-75th Percentile'
        ))

        # Add median prediction line
        fig.add_trace(go.Scatter(
            x=x,
            y=y_median,
            mode='lines',
            line=dict(color=line_color, width=3),
            name='Median Prediction'
        ))

        # Get max points
        max_median = df_view.loc[df_view['y_median'].idxmax()]
        max_upper = df_view.loc[df_view['y_upper'].idxmax()]
        max_lower = df_view.loc[df_view['y_lower'].idxmax()]


        if len(selected_year) > 1 :
            titles = f"Predicted Adjusted Profit vs Unit Price of {df_grouped['Item Name'].unique()[0]}  |  Event : {', '.join(selected_event)} in {sorted(selected_year)[0]} - {sorted(selected_year)[-1]}"
        else :
            titles = f"Predicted Adjusted Profit vs Unit Price of {df_grouped['Item Name'].unique()[0]}  |  Event : {selected_event[0]} in {sorted(selected_year)[0]}"

        # Update layout
        fig.update_layout(
            title=titles,
            xaxis_title="Unit Selling Price (RMB/kg)",
            yaxis_title="Predicted Adjusted Profit",
            template="plotly_dark"
        )

        # Add markers for 0.25 quantile predictions
        fig.add_trace(go.Scatter(
            x=df_view["Unit Selling Price (RMB/kg)"],
            y=df_view['y_lower'],
            mode='markers',
            marker=dict(color='grey', size=6),
            name='0.25 Quantile'
        ))

        # Add markers for 0.5 (median) quantile predictions
        fig.add_trace(go.Scatter(
            x=df_view["Unit Selling Price (RMB/kg)"],
            y=df_view['y_median'],
            mode='markers',
            marker=dict(color='grey', size=6),
            name='0.5 Quantile (Median)'
        ))

        # Add markers for 0.75 quantile predictions
        fig.add_trace(go.Scatter(
            x=df_view["Unit Selling Price (RMB/kg)"],
            y=df_view['y_upper'],
            mode='markers',
            marker=dict(color='grey', size=6),
            name='0.75 Quantile'
        ))

        # Add markers for highest values
        fig.add_trace(go.Scatter(
            x=[max_median['Unit Selling Price (RMB/kg)']],
            y=[max_median['y_median']],
            mode='markers+text',
            marker=dict(color='blue', size=10, symbol='circle'),
            text=[f"¥{round(max_median['y_median']):,}"],
            textposition="top center",
            name='Max Median'
        ))

        fig.add_trace(go.Scatter(
            x=[max_upper['Unit Selling Price (RMB/kg)']],
            y=[max_upper['y_upper']],
            mode='markers+text',
            marker=dict(color='green', size=10, symbol='diamond'),
            text=[f"¥{round(max_upper['y_upper']):,}"],
            textposition="top center",
            name='Max Upper'
        ))

        fig.add_trace(go.Scatter(
            x=[max_lower['Unit Selling Price (RMB/kg)']],
            y=[max_lower['y_lower']],
            mode='markers+text',
            marker=dict(color='red', size=10, symbol='square'),
            text=[f"¥{round(max_lower['y_lower']):,}"],
            textposition="bottom center",
            name='Max Lower'
        ))
        st.plotly_chart(fig)

        return max_lower,max_median,max_upper
url = "https://github.com/glenvj-j/Price-Elasticity-Supermarket-Dataset/blob/main/Dataset/clean_df.csv
response = requests.get(url)

# Load Dataset
df = pd.read_csv('response.content')
df_loss =  pd.read_csv('https://raw.githubusercontent.com/glenvj-j/Price-Elasticity-Supermarket-Dataset/refs/heads/main/Dataset/annex4.csv')
df['Event'] = df['Event'].fillna('Non-Event')

selected_year = st.sidebar.multiselect('Select Year (Can Choose 1 or more)',df['Year'].unique())

df_filtered = df[(df['Sale or Return']=='sale')&(df['Discount (Yes/No)']=='No')&(df['Year'].isin(selected_year))].copy()

df_above_100_data = df_filtered['Item Code'].value_counts().reset_index()

selected_product_code = st.sidebar.selectbox('Select Product',df_above_100_data[df_above_100_data['count']>300]['Item Code'],index=None)

try :
    st.sidebar.success(df_filtered[df_filtered['Item Code']==selected_product_code]['Item Name'].unique()[0])
except IndexError :
    st.sidebar.error('')



loss_rate_selected = df_loss.loc[df_loss['Item Code']==selected_product_code,'Loss Rate (%)']
df_selected =  df_filtered[df_filtered['Item Code']==selected_product_code][['Quantity Sold (kilo)','Unit Selling Price (RMB/kg)', 'Item Name', 'Category Name', 'Month', 'Event','Wholesale Price (RMB/kg)']]

df_grouped = (
    df_selected
    .groupby(
        ['Item Name', 'Category Name', 'Event', 'Unit Selling Price (RMB/kg)'], # add month here
        as_index=False
    )
    .agg({
        'Quantity Sold (kilo)': 'sum',
        'Wholesale Price (RMB/kg)': 'mean'
    })
)
st.sidebar.markdown(f'Total Price Point : {df_grouped.shape[0]}')
selected_event = st.sidebar.multiselect('Select Event (Can Choose 1 or more)',df_grouped['Event'].unique())

if not df_grouped.empty:
    reg_preview(df_grouped)
    
else :
    st.warning("""◄  Select your desired filters from the sidebar, then click **Show Prediction** to display the results.""")

prediction = st.sidebar.button('Show Prediction')

if prediction :
    with st.spinner("Running prediction..."):
        try :
            prediction_gam_df = model(df_grouped)
            st.balloons()
            result = show_prediction(df_grouped)
            # Create a DataFrame to show best prices
            best_prices_df = pd.DataFrame({
                "Best Price (RMB/kg)": [
                    result[2]['Unit Selling Price (RMB/kg)'],
                    result[1]['Unit Selling Price (RMB/kg)'],
                    result[0]['Unit Selling Price (RMB/kg)']
                ]
            },index=["Best Case", "Likely Case", "Worst Case"])

            # Show in Streamlit
            st.dataframe(best_prices_df.T)
        except :
            st.warning('Sample size is too small, cannot predict price elasticity')
