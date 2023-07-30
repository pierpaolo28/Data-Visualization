import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from shiny import ui, render, reactive, App
from data import inf_df, food_df, energy_df

# shiny run --reload app.py
# shinylive export . docs
# python3 -m http.server --directory docs --bind localhost 8008
# http://[::1]:8008/

def import_data(name):
    if name == "hcpi_a":
        csvStringIO = StringIO(inf_df)
        df = pd.read_csv(csvStringIO, sep=",").set_index('Date')
    elif name == "fcpi_a":
        csvStringIO = StringIO(food_df)
        df = pd.read_csv(csvStringIO, sep=",").set_index('Date')
    else:
        csvStringIO = StringIO(energy_df)
        df = pd.read_csv(csvStringIO, sep=",").set_index('Date')
    return df


inf_df = import_data("hcpi_a")
food_df = import_data("fcpi_a")
energy_df = import_data("ecpi_a")

app_ui = ui.page_fluid(
    ui.h2("Python Shiny Inflation Monitoring Application"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_selectize("country", "Country",
                list(inf_df.columns)
            ),
            ui.input_slider("range", "Years", 1970, 2022, value=(1970, 2022), step=1),
        ),
        ui.panel_main(
            ui.output_plot("overall_inflation"),
            ui.output_plot("annual_change")
        )
    ),
    ui.input_selectize("type", "Inflation Type",
        ["Food", "Energy"]
    ),
    ui.output_plot("inflation_type")
)

def server(input, output, session):
    
    @output
    @render.plot
    def overall_inflation():
        df = inf_df[input.country()].loc[inf_df[input.country()].index.isin(range(input.range()[0], input.range()[1]))]
        plt.title("Overall Inflation")
        return df.plot()
    
    @output
    @render.plot
    def annual_change():
        annual_change = inf_df[input.country()].diff().loc[inf_df[input.country()].index.isin(range(input.range()[0], input.range()[1]))]
        plt.title("Annual Change in Inflation")
        return plt.bar(annual_change.index, annual_change.values, color=np.where(annual_change>0,"Green", "Red"))
    
    @output
    @render.plot
    def inflation_type():
        if input.type() == "Food":
            df = food_df[input.country()].loc[inf_df[input.country()].index.isin(range(input.range()[0], input.range()[1]))]
            plt.title(input.country() + ' Food Inflation')
            return df.plot()
        else:
            df = energy_df[input.country()].loc[inf_df[input.country()].index.isin(range(input.range()[0], input.range()[1]))]
            plt.title(input.country() + ' Energy Inflation')
            return df.plot()


app = App(app_ui, server)