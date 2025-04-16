import streamlit as st
import joblib 
import numpy as np
import pandas as pd

model = joblib.load('outputModel.pkl')
bookingStatusEncoder = joblib.load('bookingStatsEncode.pkl')
mealPlanEncoder = joblib.load('mealPlanEncode.pkl')
roomTypeEncoder = joblib.load('roomTypeEncode.pkl')
marketSegmentEncoder = joblib.load('marketSegmentEncode.pkl')
binaryEncoder = joblib.load('binaryEncode.pkl')

def main():
    st.title('Booking Status of A Hotel Customer')

    Booking_ID = st.text_input('Your Booking ID')
    no_of_adults = st.number_input('Number of Adults', min_value = 0, step=1)
    no_of_children = st.number_input('Number of Children', min_value = 0, step=1)
    no_of_weekend_nights = st.number_input('Number of Weekend Nights', min_value = 0, step=1)
    no_of_week_nights = st.number_input('Number of Weekday Nights', min_value = 0, step=1)
    type_of_meal_plan = st.radio('Type of Meal Plan', ['Not Selected','Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3'])
    required_car_parking_space = st.radio('Require Car Parking Space', ['Yes','No'])
    room_type_reserved = st.radio('Room Type Reserved', ['Room_Type 1','Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
    lead_time = st.number_input('Lead Time', min_value = 0, step=1)
    arrival_year = st.number_input('Arrival Year', min_value = 2015, step=1)
    arrival_month = st.number_input('Arrival Month', min_value = 1, max_value = 12, step=1)
    arrival_date = st.number_input('Arrival Date', min_value = 1, max_value = 31, step=1)
    market_segment_type = st.radio('Market Segment Type', ['Online','Offline', 'Corporate', 'Complementary', 'Aviation'])
    repeated_guest = st.radio('Repeated Guest', ['Yes','No'])
    no_of_previous_cancellations = st.number_input('Amount of Bookings Cancelled', min_value = 0, step=1)
    no_of_previous_bookings_not_canceled = st.number_input('Amount of Bookings not Cancelled', min_value = 0, step=1)
    avg_price_per_room = st.number_input('Average price per Room for each Night', min_value = 0.0)
    no_of_special_requests = st.number_input('Number of Special Requests', min_value = 0, step=1)

    data = { 
        'no_of_adults' : int(no_of_adults), 
        'no_of_children' : int(no_of_children), 
        'no_of_weekend_nights' : int(no_of_weekend_nights), 
        'no_of_week_nights' : int(no_of_week_nights),
        'type_of_meal_plan' : type_of_meal_plan,
        'required_car_parking_space' : required_car_parking_space,
        'room_type_reserved' : room_type_reserved, 
        'lead_time' : int(lead_time),
        'arrival_year' : int(arrival_year), 
        'arrival_month' : int(arrival_month),
        'arrival_date'  : int(arrival_date), 
        'market_segment_type' : market_segment_type,
        'repeated_guest' : repeated_guest,
        'no_of_previous_cancellations' : int(no_of_previous_cancellations), 
        'no_of_previous_bookings_not_canceled' : int(no_of_previous_bookings_not_canceled), 
        'avg_price_per_room' : float(avg_price_per_room),
        'no_of_special_requests' : int(no_of_special_requests)
    }

    df = pd.DataFrame([list(data.values())], columns=['no_of_adults',
                                                      'no_of_children', 
                                                      'no_of_weekend_nights', 'no_of_week_nights',
                                                      'type_of_meal_plan',
                                                      'required_car_parking_space',
                                                      'room_type_reserved', 
                                                      'lead_time',
                                                      'arrival_year',
                                                      'arrival_month',
                                                      'arrival_date', 
                                                      'market_segment_type',
                                                      'repeated_guest','no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'avg_price_per_room',
                                                      'no_of_special_requests'])
    
    df = df.replace(binaryEncoder)

    mealPlanEnc = df[['type_of_meal_plan']]
    roomTypeEnc = df[['room_type_reserved']]
    marketSegmentEnc = df[['market_segment_type']]

    mealPlanEnc = pd.DataFrame(mealPlanEncoder.transform(mealPlanEnc).toarray(), columns=mealPlanEncoder.get_feature_names_out())
    roomTypeEnc = pd.DataFrame(roomTypeEncoder.transform(roomTypeEnc).toarray(), columns=roomTypeEncoder.get_feature_names_out())
    marketSegmentEnc = pd.DataFrame(marketSegmentEncoder.transform(marketSegmentEnc).toarray(), columns=marketSegmentEncoder.get_feature_names_out())

    df = pd.concat([df, mealPlanEnc, roomTypeEnc, marketSegmentEnc], axis=1)
    df = df.drop(columns = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'], axis=1)

    

    if st.button('Get Booking Status'):
        # st.dataframe(df)
        inputData = df      
        result = predictResult(inputData)
        result = bookingStatusEncoder.inverse_transform(result)
        st.success(f'Your Booking Status: {result[0]}')

def predictResult(inputData):
    inputArray = np.array(inputData).reshape(1, -1)
    predictResult = model.predict(inputArray)
    # st.dataframe(predictResult)
    return predictResult

if __name__ == '__main__':
    main()
