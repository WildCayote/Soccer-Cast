import pandas as pd
import numpy as np
import os , math 

from src.utils.reading import read_data


class TranformData:
    '''
    This class will accept the folder of a league and then preps and compiles all of the leagues seasons into one pandas data frame.
    '''
          
    def __init__(self , folder_path : str):
        self.league_path = folder_path
        self.league_seasons = self.__get_seasons()
        self.total_games = 0
        self.expected_games = 0
        self.total_season = len(self.league_seasons)
        self.data = pd.DataFrame()

    def __get_seasons(self):
        seasons = os.listdir(self.league_path)
        return [os.path.join(self.league_path , season) for season in seasons]
    
    def __read_data(self , data_path : str):
        return read_data(data_path=data_path)
    
    def calculate_expected_result(self , home_elo : float , away_elo : float , home_ground_advantage : float = 100.0 , scaling_factor : float = 400):
        # calculate expected outcome for the home_team
        elo_difference = (home_elo - away_elo + home_ground_advantage) / scaling_factor
        denomenator = 1 + math.pow(math.e , elo_difference * -1)
        home_exp_outcome = 1 / denomenator

        # calculate expected outcome for the away_team
        elo_difference = (away_elo - home_elo) / scaling_factor
        denomenator = 1 + math.pow(math.e , elo_difference * -1)
        away_exp_outcome = 1 / denomenator

        return (home_exp_outcome , away_exp_outcome)
    
    def calculate_elo(self, home_init : float , away_init : float , home_exp : float , away_exp : float , match_result : int , k : float = 100):
    
        # convert the match result to perespective result for home/away team
        # keep in mind match_result is one of (1 , 2 , 3) corresponding to (home_win , away_win , draw)
        # we want to map these outcome to (1 , 0.5 , 0) for each team which corresponds to (win , draw , loss)         
        if match_result == 1:
            home_result = 1
            away_result = 0
        elif match_result == 2:
            home_result = 0
            away_result = 1
        else:
            home_result = 0.5 
            away_result = 0.5

        # calculate the new elo of the home team
        home_elo = home_init + k * (home_result - home_exp)

        # calculate the new elo of the away team
        away_elo = away_init + k * (away_result - away_exp)

        return (home_elo , away_elo)

    def poisson(self , exp : float, target : int):
        '''
        Implementation of the poisson equation

        Args:
            - exp : float , the expected value
            - target : a positive int , the target value

        Returns:
            - float , a probability value within [0,1]
        '''

        nominator = math.pow(math.e , -exp) * math.pow(exp ,target) 
        denominator = math.factorial(target)

        return nominator / denominator

    def poisson_dist(self, home_exp : float , away_exp : float , max_goals : float = 13):
        '''
        Accepts the home and away expected goals and then returns the poisson distribution for each team.

        Args:
            - home_exp : float , home team's expected goals
            - away_exp : float , away team's expected goals
            - max_goals : int , the max amount of goals to calculate poisson distribution for

        Returns:
            - A numpy array of shape (2 , max_goals + 1) , the first row represents the goal distribution of the home team (from 0 - 9) and the second row represents the goal distribution of the away team (from 0 - 9)
        '''

        # numpy array for holding the output
        dist = np.zeros(shape=(2 , max_goals + 1))

        for goal in range(max_goals + 1):

            # calculate the probability of scoring the goal for home and away team
            goal_prob_home = self.poisson(exp=home_exp , target=goal)
            goal_prob_away = self.poisson(exp=away_exp , target=goal)

            # add the home prob result to the final array
            dist[0][goal] = goal_prob_home

            # add the away prob result to the final array
            dist[1][goal] = goal_prob_away

        return dist

    def dist_to_prediction(self , dist : np.ndarray):
        '''
        Accepts the poisson distribution of a game and the inferce the outcomes of the match. The outcomes are:
            - home win
            - away win
            - home double chance
            - away double chance
            - draw / no draw
            - over / under 0.5
            - over / under 1.5
            - over / under 2.5
            - over / under 3.5

        Args:
            - dist : np.ndarray , the poisson distribution of a match

        Returns:
            - A dictionary which contains the name of the outcomes as the keys and the probability of the outcome as the values
        '''

        predictions = {}


        # find the probabilit of drawing and not drawing
        draw = np.dot(dist[0] , dist[1])
        predictions['x'] = draw
        predictions['12'] = 1 -draw

        # calculate home win and away win
        home_win = 0
        away_win = 0

        for winner_goal in range(1 , dist.shape[1]):
            for loosing_goal in range(0 , winner_goal):
                # home win 
                home_win += dist[0][winner_goal] * dist[1][loosing_goal]

                # away win
                away_win += dist[1][winner_goal] * dist[0][loosing_goal]

        predictions['1'] = home_win
        predictions['2'] = away_win

        # calculate double chance
        predictions['1x'] = home_win + draw
        predictions['2x'] = away_win + draw

        # calculate probability of all goal combinations
        all_results = np.outer(dist[0] , dist[1])

        # add over/under 0.5
        predictions['over0.5'] = 1 - all_results[0][0]
        predictions['under0.5'] = all_results[0][0]

        # add over/under 1.5
        all_under_1_5 = [(0,0) , (1,0) , (0,1)]
        probs = [all_results[val[0]][val[1]] for val in all_under_1_5 ]
        under1_5= np.array(probs).sum()
        predictions['under1.5'] = under1_5
        predictions['over1.5'] = 1 - under1_5

        # add over/under 2.5
        all_under_2_5 = [(2,0) , (0,2) , (1,1)] 
        probs = [all_results[val[0]][val[1]] for val in all_under_2_5]
        under2_5 = np.array(probs).sum() + under1_5
        predictions['under2.5'] = under2_5
        predictions['over2.5'] = 1 - under2_5

        # add over/under 3.5
        all_under_3_5 = [(3,0) , (0,3) , (2,1) , (1,2)]
        probs = [all_results[val[0]][val[1]] for val in all_under_3_5]
        under3_5 = np.array(probs).sum() + under2_5
        predictions['under3.5'] = under3_5
        predictions['over3.5'] = 1 - under3_5

        # add over/under 4.5
        all_under_4_5 = [(3,1) , (1,3) , (2,2) , (4,0) , (0,4)]
        probs = [all_results[val[0]][val[1]] for val in all_under_4_5]
        under4_5 = np.array(probs).sum() + under3_5
        predictions['under4.5'] = under4_5
        predictions['over4.5'] = 1 - under4_5
        return predictions

    def match_outcomes(self , home_score : int , away_score : int):
        '''
        A function for determining what the outcomes of the game were from the score line.
        i.e returns a dictionary determining whether the game was a draw , over/under 0.5 and the like

        Args:
            - home_score : int , the amount of goals scored by the home team
            - away_score : int , the amount of goals scored by the away team

        Returns:
            - A dictionary which contains the names of the outcomes as keys and the value either 1 or 0 indicating True or False respectively
        '''

        match_state = {}

        # goal state
        goals_sum = home_score + away_score

        under_0_5 = 1 if goals_sum < 0.5 else 0
        under_1_5 = 1 if goals_sum < 1.5 else 0
        under_2_5 = 1 if goals_sum < 2.5 else 0
        under_3_5 = 1 if goals_sum < 3.5 else 0
        under_4_5 = 1 if goals_sum < 4.5 else 0

        match_state['under0.5'] = under_0_5
        match_state['under1.5'] = under_1_5
        match_state['under2.5'] = under_2_5
        match_state['under3.5'] = under_3_5
        match_state['under4.5'] = under_4_5

        # match state
        draw = home_score == away_score
        home_win = 1 if home_score > away_score else 0
        away_win = 1 if home_score < away_score else 0
        home_double = 1 if home_win == 1 or draw else 0
        away_double = 1 if away_win == 1 or draw else 0

        match_state['draw'] = 1 if draw else 0
        match_state['home_win'] = home_win
        match_state['away_win'] = away_win
        match_state['home_double'] = home_double
        match_state['away_double'] = away_double

        return match_state

    def _load_season(self , season_path : str):
        # read the data
        df = self.__read_data(data_path=season_path)
        
        # get the amount of games collected in the season so far
        collected_games = df.shape[0]

        # get all the teams
        home_teams = df['home_team'].unique()
        away_teams = df['away_team'].unique()

        teams = np.array([*home_teams , *away_teams])
        teams = np.unique(teams)

        number_teams = len(teams)

        # calculate the number of expected games in a season using permutation
        expected_games = math.perm(number_teams , 2)

        # calculate the games per week
        games_per_week = number_teams // 2

        return {
            'games_per_week' : games_per_week,
            'expected_games' : expected_games,
            'collected_games' : collected_games,
            'teams' : teams,
            'df' : df
        }

    def __prep_season(self , games_per_week : int , collected_games : int , teams : list , df : pd.DataFrame , expected_games : int):

        # calculated the number of gameweeks
        num_gws = math.ceil(collected_games / games_per_week)

        # split the whole data into game_weeks
        game_weeks = np.array_split(df , num_gws)

        # create a dictionary for holding attack_strength , defence_strength , league_home_score/concede , league_away_score/concede 
        gw_zero_stats = {'home_score' : 0 , 'away_score' : 0 , 'home_concede' : 0 , 'away_concede' : 0 , 'home_games': 0 , 'away_games' : 0 , 'home_attack_strength' : 0 , 'home_defence_strength' : 0 , 'away_attack_strength' : 0 ,'away_defence_strength' : 0 , 'elo_rating' : 1000}
        team_stat_history = {'teams_stats' : {team : { 0 : gw_zero_stats.copy()} for team in teams}}
        team_stat_history.update({'league_stats' : {0 : {'home_score' : 0 , 'away_score' : 0 , 'home_concede' : 0 , 'away_concede' : 0 , 'num_games' : 0 , 'home_score_avg' : 0 , 'away_score_avg' : 0}}})

        # current_gameweek
        current_gw = 0

        # dictionary for holding the new columns to be added to the dataframe
        new_data = {
            'home_attack_strength' : [] , 
            'home_defence_strength' : [] , 
            'away_attack_strength' : [] , 
            'away_defence_strength' : [] , 
            'home_score_avg' : [] , 
            'away_score_avg' : [] , 
            'home_expected_goal' : [] , 
            'away_expected_goal' : [] , 
            'home_elo' : [] , 
            'away_elo' : [] , 
            '1' : [] , 
            '2' :[] ,  
            'x' : [],
            '12' : [],
            '1x' : [],
            '2x' : [],
            'ov0.5' : [],
            'un0.5' : [],
            'ov1.5' : [],
            'un1.5' : [],
            'ov2.5' : [],
            'un2.5' : [],
            'ov3.5' : [],
            'un3.5' : [],
            'ov4.5' : [],
            'un4.5' : [],
            'num_home_games' : [] , 
            'num_away_games' : [] ,
            'home_win' : [],
            'away_win' :[],
            'home_double' : [],
            'away_double' : [],
            'under0.5' : [],
            'under1.5' : [],
            'under2.5' : [],
            'under3.5' : [],
            'under4.5' : [],
            'draw' : []
        }

        for game_week in game_weeks:
            # the game_week read into a dataframe
            gw_df = pd.DataFrame(game_week)

            # calculate the total goal scored (home/away)
            home_scored = gw_df['home_score'].sum()
            away_scored = gw_df['away_score'].sum()

            # get the number of games in this dataframe and the previous one
            num_games = gw_df.shape[0]
            prev_games = team_stat_history['league_stats'][current_gw]['num_games']
            prev_home_score = team_stat_history['league_stats'][current_gw]['home_score']
            prev_away_score = team_stat_history['league_stats'][current_gw]['away_score']

            current_tot_games = prev_games + num_games

            # add the new league stats for the next game week , i.e current_gw + 1
            team_stat_history['league_stats'][current_gw + 1] = {'home_score' : prev_home_score + home_scored , 'away_score' : prev_away_score + away_scored , 'home_concede' : prev_away_score + away_scored , 'away_concede' : prev_home_score + home_scored , 'num_games' : current_tot_games , 'home_score_avg' : (prev_home_score + home_scored) / current_tot_games , 'away_score_avg' :  (prev_away_score + away_scored) / current_tot_games}

            # loop throught the dataframe (the matches)
            home_teams = gw_df['home_team']
            for home_team in home_teams:
            
                game = gw_df[gw_df['home_team'] == home_team]
                away_team = game['away_team'].tolist()[0]

                home_current_gw = list(team_stat_history['teams_stats'][home_team].keys())[-1]
                away_current_gw = list(team_stat_history['teams_stats'][away_team].keys())[-1]

                game_home_score = game['home_score'].tolist()[0]
                game_away_score = game['away_score'].tolist()[0]


                team_stat_history['teams_stats'][away_team][away_current_gw + 1] = {}
                team_stat_history['teams_stats'][home_team][home_current_gw + 1] = {}

                # update the teams stats for the ones playing at home             
                team_stat_history['teams_stats'][home_team][home_current_gw + 1].update({'home_score' : float(team_stat_history['teams_stats'][home_team][home_current_gw]['home_score'] +  game_home_score)})
                team_stat_history['teams_stats'][home_team][home_current_gw + 1].update({'home_concede' : float(team_stat_history['teams_stats'][home_team][home_current_gw]['home_concede'] + game_away_score)})
                team_stat_history['teams_stats'][home_team][home_current_gw + 1].update({'home_games' : float(team_stat_history['teams_stats'][home_team][home_current_gw]['home_games'] + 1)})
                home_score_avg = team_stat_history['teams_stats'][home_team][home_current_gw + 1]['home_score'] / team_stat_history['teams_stats'][home_team][home_current_gw + 1]['home_games']
                home_concede_avg = team_stat_history['teams_stats'][home_team][home_current_gw + 1]['home_concede'] / team_stat_history['teams_stats'][home_team][home_current_gw + 1]['home_games']
                team_stat_history['teams_stats'][home_team][home_current_gw + 1].update({'home_attack_strength' : float(home_score_avg / team_stat_history['league_stats'][current_gw + 1]['home_score_avg'])})
                team_stat_history['teams_stats'][home_team][home_current_gw + 1].update({'home_defence_strength' : float(home_concede_avg / team_stat_history['league_stats'][current_gw + 1]['away_score_avg'])})
                team_stat_history['teams_stats'][home_team][home_current_gw + 1].update({'away_score' : team_stat_history['teams_stats'][home_team][home_current_gw]['away_score']})
                team_stat_history['teams_stats'][home_team][home_current_gw + 1].update({'away_concede' : team_stat_history['teams_stats'][home_team][home_current_gw]['away_concede']})
                team_stat_history['teams_stats'][home_team][home_current_gw + 1].update({'away_games' : team_stat_history['teams_stats'][home_team][home_current_gw]['away_games']})
                team_stat_history['teams_stats'][home_team][home_current_gw + 1].update({'away_attack_strength' : team_stat_history['teams_stats'][home_team][home_current_gw]['away_attack_strength']})
                team_stat_history['teams_stats'][home_team][home_current_gw + 1].update({'away_defence_strength' : team_stat_history['teams_stats'][home_team][home_current_gw]['away_defence_strength']})

                # update the teams stats for the ones playing away
                team_stat_history['teams_stats'][away_team][away_current_gw + 1].update({'away_score' : float(team_stat_history['teams_stats'][away_team][away_current_gw]['away_score'] +  game_away_score)})
                team_stat_history['teams_stats'][away_team][away_current_gw + 1].update({'away_concede' : float(team_stat_history['teams_stats'][away_team][away_current_gw]['away_concede'] + game_home_score)})
                team_stat_history['teams_stats'][away_team][away_current_gw + 1].update({'away_games' : float(team_stat_history['teams_stats'][away_team][away_current_gw]['away_games'] + 1)})
                away_score_avg = team_stat_history['teams_stats'][away_team][away_current_gw + 1]['away_score'] / team_stat_history['teams_stats'][away_team][away_current_gw + 1]['away_games']
                away_concede_avg = team_stat_history['teams_stats'][away_team][away_current_gw + 1]['away_concede'] / team_stat_history['teams_stats'][away_team][away_current_gw + 1]['away_games']
                team_stat_history['teams_stats'][away_team][away_current_gw + 1].update({'away_attack_strength' : float(away_score_avg / team_stat_history['league_stats'][current_gw + 1]['away_score_avg'])})
                team_stat_history['teams_stats'][away_team][away_current_gw + 1].update({'away_defence_strength' : float(away_concede_avg / team_stat_history['league_stats'][current_gw + 1]['home_score_avg'])})
                team_stat_history['teams_stats'][away_team][away_current_gw + 1].update({'home_score' : team_stat_history['teams_stats'][away_team][away_current_gw]['home_score']})
                team_stat_history['teams_stats'][away_team][away_current_gw + 1].update({'home_concede' : team_stat_history['teams_stats'][away_team][away_current_gw]['home_concede']})
                team_stat_history['teams_stats'][away_team][away_current_gw + 1].update({'home_games' : team_stat_history['teams_stats'][away_team][away_current_gw]['home_games']})
                team_stat_history['teams_stats'][away_team][away_current_gw + 1].update({'home_attack_strength' : team_stat_history['teams_stats'][away_team][away_current_gw]['home_attack_strength']})
                team_stat_history['teams_stats'][away_team][away_current_gw + 1].update({'home_defence_strength' : team_stat_history['teams_stats'][away_team][away_current_gw]['home_defence_strength']})

                # update the teams elo as a result of the current game
                prev_home_elo = team_stat_history['teams_stats'][home_team][home_current_gw]['elo_rating']
                prev_away_elo = team_stat_history['teams_stats'][away_team][away_current_gw]['elo_rating']
                match_result = game['result'].tolist()[0]

                # calculate the expected outcome of the game based on the pre_match elo
                home_exp_outcome , away_exp_outcome = self.calculate_expected_result(home_elo=prev_home_elo , away_elo=prev_away_elo)

                # calculate the new elo as a result of the match result and update it for the next time
                new_home_elo , new_home_elo = self.calculate_elo(home_init=prev_home_elo , away_init=prev_away_elo , home_exp=home_exp_outcome , away_exp=away_exp_outcome , match_result=match_result)
                team_stat_history['teams_stats'][home_team][home_current_gw + 1].update({'elo_rating' : new_home_elo})
                team_stat_history['teams_stats'][away_team][away_current_gw + 1].update({'elo_rating' : new_home_elo})

                # add the home/away attack_strength , home/away defence_strength columns to the data frame 
                home_att_str = team_stat_history['teams_stats'][home_team][home_current_gw]['home_attack_strength']
                home_def_str = team_stat_history['teams_stats'][home_team][home_current_gw]['home_defence_strength']
                away_att_str = team_stat_history['teams_stats'][away_team][away_current_gw]['away_attack_strength']
                away_def_str = team_stat_history['teams_stats'][away_team][away_current_gw]['away_defence_strength']
                home_scr_avg = team_stat_history['league_stats'][current_gw]['home_score_avg']
                away_scr_avg = team_stat_history['league_stats'][current_gw]['away_score_avg']
                home_exp_goal = float(home_att_str * away_def_str * home_scr_avg)
                away_exp_goal = float(away_att_str * home_def_str * away_scr_avg)

                # number of games played at home by the home team
                num_home_games = team_stat_history['teams_stats'][home_team][home_current_gw]['home_games']

                # number of games played away by the away team
                num_away_games = team_stat_history['teams_stats'][away_team][away_current_gw]['away_games']
                
                # calculate the poisson distribution probabilities for 'x' , '12' , '1' , '2' , '1x' , '2x' , 'ov0.5' , 'un0.5 , 'un1.5', 'ov1.5' , 'un2.5' , 'ov2.5' , 'un3.5' , 'ov3.5' , 'un4.5' , 'ov4.5' 
                dist = self.poisson_dist(home_exp=home_exp_goal , away_exp=away_exp_goal)
                event_probs = self.dist_to_prediction(dist=dist) 

                new_data['1'].append(event_probs['1'])
                new_data['2'].append(event_probs['2'])
                new_data['x'].append(event_probs['x'])
                new_data['1x'].append(event_probs['1x'])
                new_data['2x'].append(event_probs['2x'])
                new_data['12'].append(event_probs['12'])
                new_data['ov0.5'].append(event_probs['over0.5'])
                new_data['un0.5'].append(event_probs['under0.5'])
                new_data['ov1.5'].append(event_probs['over1.5'])
                new_data['un1.5'].append(event_probs['under1.5'])
                new_data['ov2.5'].append(event_probs['over2.5'])
                new_data['un2.5'].append(event_probs['under2.5'])
                new_data['ov3.5'].append(event_probs['over3.5'])
                new_data['un3.5'].append(event_probs['under3.5'])
                new_data['ov4.5'].append(event_probs['over4.5'])
                new_data['un4.5'].append(event_probs['under4.5'])

                # add the new match events infered from the score line
                match_events = self.match_outcomes(home_score=game_home_score , away_score=game_away_score)
                new_data['home_win'].append(match_events['home_win'])
                new_data['away_win'].append(match_events['away_win'])
                new_data['home_double'].append(match_events['home_double'])
                new_data['away_double'].append(match_events['away_double'])
                new_data['under0.5'].append(match_events['under0.5'])
                new_data['under1.5'].append(match_events['under1.5'])
                new_data['under2.5'].append(match_events['under2.5'])
                new_data['under3.5'].append(match_events['under3.5'])
                new_data['under4.5'].append(match_events['under4.5'])
                new_data['draw'].append(match_events['draw'])
                new_data['home_attack_strength'].append(home_att_str)
                new_data['home_defence_strength'].append(home_def_str)
                new_data['away_attack_strength'].append(away_att_str)
                new_data['away_defence_strength'].append(away_def_str)
                new_data['home_score_avg'].append(home_scr_avg)
                new_data['away_score_avg'].append(away_scr_avg)
                new_data['home_expected_goal'].append(home_exp_goal)
                new_data['away_expected_goal'].append(away_exp_goal)
                new_data['num_home_games'].append(num_home_games)
                new_data['num_away_games'].append(num_away_games)
                new_data['home_elo'].append(prev_home_elo)
                new_data['away_elo'].append(prev_away_elo)
                current_gw_stats = team_stat_history['league_stats'][current_gw]
            
            # update the counter
            current_gw += 1
        
        # the data_frame that contains the new features engineered from the old ones
        preped_dataframe = pd.DataFrame.from_dict(new_data)
        
        # combining the old features and new features to create a new dataframe
        df = df.join(preped_dataframe, lsuffix='_left', rsuffix='_right')

        return df
    
    def transform(self):
        # loop through the seasons 
        for season in self.league_seasons:

            # load the season 
            loaded_season = self._load_season(season_path=season)

            # update the leagues metadata
            self.total_games += loaded_season['collected_games']
            self.expected_games += loaded_season['expected_games']

            # prep the season data
            preped_season = self.__prep_season(**loaded_season)

            self.data = pd.concat([self.data , preped_season])
        
        return self.data



if __name__ == '__main__':
    league_path = "C:\\Users\\dan\\Desktop\\Projects\\Predictor\\scraped_data\\leagues_old\\18"
    preprocessor = TranformData(league_path)
    preped_data = preprocessor.transform()