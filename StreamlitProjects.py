import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def Electricity_Cost():
    st.title("🏠 Electric Fitting Cost Estimator")
    st.markdown("Estimate your house's electric fitting cost based on number of rooms, size, and wire thickness.")

    COST_PER_LIGHT = 150
    COST_PER_FAN = 1200
    COST_PER_SWITCH = 40
    COST_PER_SOCKET = 120
    COST_PER_MCB = 500

    WIRE_COST = {
        "1.0 mm": 20,
        "1.5 mm": 30,
        "2.5 mm": 40,
        "4.0 mm": 60
    }

    st.sidebar.header("🏠 Room Details")
    num_rooms = st.sidebar.number_input("Number of Rooms", min_value=1, value=2)
    num_halls = st.sidebar.number_input("Number of Halls", min_value=0, value=1)
    room_area = st.sidebar.number_input("Average Area of a Room (sq. ft.)", min_value=50, value=120)
    hall_area = st.sidebar.number_input("Average Area of a Hall (sq. ft.)", min_value=100, value=180)
    wire_size = st.sidebar.selectbox("Select Wire Thickness", options=list(WIRE_COST.keys()))

    total_area = (num_rooms * room_area) + (num_halls * hall_area)
    lights = int(total_area // 50)
    fans = int(total_area // 120)
    switches = lights + fans
    sockets = int(total_area // 100)
    mcb = 1
    wiring_length = int((total_area / 100) * 10)

    total_cost = (
        lights * COST_PER_LIGHT +
        fans * COST_PER_FAN +
        switches * COST_PER_SWITCH +
        sockets * COST_PER_SOCKET +
        mcb * COST_PER_MCB +
        wiring_length * WIRE_COST[wire_size]
    )

    st.subheader("📊 Cost Estimation Summary")
    st.write(f"🧱 Total Area: {total_area} sq. ft.")
    st.write(f"🔌 Estimated Lights: {lights}")
    st.write(f"🌀 Estimated Fans: {fans}")
    st.write(f"🎚️ Estimated Switches: {switches}")
    st.write(f"🔌 Estimated Power Sockets: {sockets}")
    st.write(f"🧯 Wire Length Required: {wiring_length} meters ({wire_size})")
    st.write(f"📦 MCB Box: {mcb}")
    st.markdown("---")
    st.success(f"💰 **Total Estimated Cost: ₹{total_cost}**")

    with st.expander("📌 Tips"):
        st.write("- Always keep 10–15% extra wiring for flexibility.")
        st.write("- For kitchen and AC, use 2.5 mm or 4 mm wire.")
        st.write("- Confirm socket placements with an electrician.")


def Mark_Pre():
    st.title("📚 Study Hours vs Marks Predictor")
    st.markdown("Predict your marks based on study hours using linear regression")

    @st.cache_data
    def load_data():
        data = {
            'Hours': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7],
            'Marks': [21, 47, 27, 75, 30, 20, 88, 60, 81, 25]
        }
        return pd.DataFrame(data)

    df = load_data()
    with st.expander("View Raw Data"):
        st.dataframe(df)

    model = LinearRegression()
    model.fit(df[['Hours']], df['Marks'])

    hours = st.sidebar.number_input("Enter study hours:", min_value=0.5, max_value=24.0, value=5.0, step=0.5)
    if st.sidebar.button("Predict Marks"):
        prediction = model.predict([[hours]])[0]
        st.success(f"Predicted Marks: {prediction:.1f}")

    st.header("Model Information")
    col1, col2 = st.columns(2)
    col1.metric("Coefficient (Slope)", f"{model.coef_[0]:.2f}")
    col2.metric("Intercept", f"{model.intercept_:.2f}")
    st.markdown(f"*Regression Equation:* Marks = {model.coef_[0]:.2f} × Hours + {model.intercept_:.2f}")

    st.header("Data Visualization")
    st.scatter_chart(df.set_index('Hours'))
    x_values = np.linspace(df['Hours'].min(), df['Hours'].max(), 100).reshape(-1, 1)
    y_values = model.predict(x_values)
    st.line_chart(pd.DataFrame({'Predicted Marks': y_values.flatten()}, index=x_values.flatten()))


def PG_RentPr():
    st.title("🏠 PG Price Predictor")
    st.write("Predict the monthly PG price based on room sharing, AC, and food.")

    data = {
        'Persons': [1, 2, 2, 3, 3, 1, 1, 2, 3],
        'AC': [1, 1, 0, 1, 0, 0, 1, 0, 1],
        'Food': [1, 0, 1, 1, 0, 1, 1, 0, 1],
        'Price': [9000, 6500, 5500, 5000, 4000, 8500, 9500, 5200, 4800]
    }
    df = pd.DataFrame(data)
    X = df[['Persons', 'AC', 'Food']]
    y = df['Price']
    model = LinearRegression().fit(X, y)

    st.sidebar.header("📝 Your Room Preferences")
    persons = st.sidebar.selectbox("Number of Persons in Room", [1, 2, 3])
    ac_room = st.sidebar.radio("Room Type", ["AC", "Non-AC"])
    food = st.sidebar.radio("Food Included", ["Yes", "No"])

    ac_flag = 1 if ac_room == "AC" else 0
    food_flag = 1 if food == "Yes" else 0

    predicted_price = model.predict([[persons, ac_flag, food_flag]])[0]

    st.subheader("💰 Estimated Monthly PG Rent")
    st.success(f"For a {persons}-person {'AC' if ac_flag else 'Non-AC'} room with{'out' if not food_flag else ''} food: ₹{predicted_price:.2f}")

    with st.expander("📂 Show Sample Training Data"):
        st.dataframe(df)

    with st.expander("📈 Model Equation"):
        coef = model.coef_
        intercept = model.intercept_
        st.code(f"Price = {coef[0]:.2f} × Persons + {coef[1]:.2f} × AC + {coef[2]:.2f} × Food + {intercept:.2f}")


def Student_guid():
    st.title("Student Career Guidance by Subject")

    guidance = {
        "math": ["Engineer", "Data Scientist", "Mathematician", "Actuary", "Economist"],
        "science": ["Doctor", "Pharmacist", "Research Scientist", "Biotechnologist", "Environmentalist"],
        "physics": ["Mechanical Engineer", "Physicist", "Astronomer", "Robotics Engineer"],
        "chemistry": ["Chemical Engineer", "Pharmacologist", "Forensic Scientist", "Material Scientist"],
        "biology": ["Doctor", "Geneticist", "Zoologist", "Microbiologist"],
        "computer": ["Software Engineer", "Web Developer", "AI/ML Engineer", "Cybersecurity Expert"],
        "english": ["Journalist", "Content Writer", "Teacher", "Editor"],
        "history": ["Historian", "Archaeologist", "Civil Services", "Museum Curator"],
        "geography": ["Geologist", "Urban Planner", "Cartographer", "Environmental Consultant"],
        "commerce": ["CA (Chartered Accountant)", "Banker", "Business Analyst", "Financial Advisor"],
        "arts": ["Designer", "Animator", "Musician", "Fine Artist"]
    }

    subject = st.text_input("Enter your favorite subject:")

    if subject:
        subject_lower = subject.lower()
        if subject_lower in guidance:
            st.write(f"### 📘 Based on your interest in {subject.capitalize()}, you can explore:")
            for career in guidance[subject_lower]:
                st.write(f"- {career}")
        else:
            st.warning("⚠️ Sorry, guidance for this subject is not available.")
            st.info("Try subjects like: Math, Science, Physics, Computer, etc.")


# Main App

st.set_page_config(page_title="Multi Tool App", layout="wide")
st.sidebar.title("Select Tool")

app_mode = st.sidebar.radio("Go to", 
    ("Electricity Cost Estimator", "Marks Predictor", "PG Rent Predictor", "Student Career Guidance"))

if app_mode == "Electricity Cost Estimator":
    Electricity_Cost()
elif app_mode == "Marks Predictor":
    Mark_Pre()
elif app_mode == "PG Rent Predictor":
    PG_RentPr()
elif app_mode == "Student Career Guidance":
    Student_guid()
