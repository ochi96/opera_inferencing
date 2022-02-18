def baseline_model():
    model = Sequential()
    model.add(Dense(23, input_dim=23, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(500, kernel_initializer='normal', activation='relu'))
    model.add(Dense(500, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation="relu"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train_neural_net() -> None:
    pd.set_option('display.max_columns', None)
    df = pd.read_pickle("cleaned_data.pkl")
    df["cap_percentage"].fillna(0, inplace=True)
    x_df = df.drop(["target_cap_percentage", "target_salary", "player_id", "date", "current_salary", "salary_cap"],
                   axis=1)
    x = np.asarray(x_df).astype('float32')
    y = df["target_cap_percentage"].to_numpy().reshape(-1, 1)
    estimators = []
    estimators.append(('standardize', MinMaxScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=16, verbose=True)))
    pipeline = Pipeline(estimators)
    pipeline.fit(x, y)
    output = pipeline.predict(x)
    predictions = pd.DataFrame()
    predictions["Salary Cap"] = x_df["target_salary_cap"]
    predictions["Predictions"] = output
    predictions["Predictions"] = predictions["Predictions"] * predictions["Salary Cap"]
    predictions["Actuals"] = y
    predictions["Actuals"] = predictions["Actuals"] * predictions["Salary Cap"]
    predictions.to_csv("Predictions.csv")
    # pipeline.named_steps["mlp"].model.save("keras_model.h5")
    # pipeline.named_steps['mlp'].model = None
    # joblib.dump(pipeline, 'nba_pipeline.pkl')