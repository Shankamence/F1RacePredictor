import requests
import pandas as pd


def fetch_race_results(season, start_round, end_round):
    race_data = []

    for round_number in range(start_round, end_round + 1):
        url = f'https://ergast.com/api/f1/{season}/{round_number}/results.json'
        response = requests.get(url)
        data = response.json()  # Parse response as JSON

        if 'Races' in data['MRData']['RaceTable']:
            race = data['MRData']['RaceTable']['Races'][0]  # Accessing the first race in the list

            for result in race['Results']:
                race_data.append({
                    'season': race['season'],
                    'round': race['round'],
                    'raceName': race['raceName'],
                    'circuitName': race['Circuit']['circuitName'],
                    'circuitLocation': race['Circuit']['Location']['locality'],
                    'circuitCountry': race['Circuit']['Location']['country'],
                    'driverId': result['Driver']['driverId'],
                    'driverName': f"{result['Driver']['givenName']} {result['Driver']['familyName']}",
                    'constructorId': result['Constructor']['constructorId'],
                    'constructorName': result['Constructor']['name'],
                    'gridPosition': int(result['grid']),
                    'finishPosition': int(result['position']),
                    'points': float(result['points']),
                    'laps': int(result['laps']),
                    'raceStatus': result['status'],
                    'raceTimeMillis': int(result['Time']['millis']) if 'Time' in result else None,
                    'raceTime': result['Time']['time'] if 'Time' in result else None,
                    'fastestLapTime': result['FastestLap']['Time']['time'] if 'FastestLap' in result else None,
                    'fastestLapRank': int(result['FastestLap']['rank']) if 'FastestLap' in result else None,
                    'averageSpeedKph': float(
                        result['FastestLap']['AverageSpeed']['speed']) if 'FastestLap' in result else None
                })

    # Convert the list of race data into a Pandas DataFrame
    df = pd.DataFrame(race_data)

    # Save the DataFrame to a CSV file in the data directory
    csv_filename = f'f1_race_results_{season}_rounds_{start_round}_to_{end_round}.csv'
    df.to_csv(csv_filename, index=False)

    print(f"Data collection complete. Data saved to {csv_filename}")


if __name__ == "__main__":
    # Example: Fetch results for season 2024, rounds 1 to 15
    fetch_race_results(2024, 1, 15)
