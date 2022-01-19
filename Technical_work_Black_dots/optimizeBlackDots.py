#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import numpy as np


class OptimizeBlackDots():
    """Optimization of position of black dots

    Class with functions for modifying the positions of black dots
    for PFS spectrograph
    """

    def __init__(self, dots, list_of_mcs_data_all, list_of_descriptions):
        """
        Parameters
        ----------
        dots : `pandas.dataframe`, (N_fiber, 4)
            Dataframe contaning initial guesses for dots
            The columns are ['spotId', 'x', 'y', 'r']
        list_of_mcs_data_all : `list` of `np.arrays`
            List contaning arrays with x and y positions
            of moving cobra from mcs observations
            Each array with a shape (2, N_fiber, N_obs)
        list_of_descriptions : `list` of `str`
            What kind of moves have been made (theta or phi)

        Examples
        ----------
        # example for November 2021 run to create
        # list_of_mcs_data_all and list_of_descriptions
        # which are inputs for the class

        >>> list_of_mcs_data_all = [mcs_data_all_1, mcs_data_all_2]
        >>> list_of_descriptions = ['theta', 'phi']
        """
        self.dots = dots
        self.n_of_obs = len(list_of_descriptions)
        self.number_of_fibers = 2394
        # radius of black dots is 0.75 mm
        # small variations exist, but those have neglible impact
        self.radius_of_black_dots = 0.75

        obs_and_predict_multi = []
        for obs in range(len(list_of_mcs_data_all)):
            mcs_data_all = list_of_mcs_data_all[obs]
            description = list_of_descriptions[obs]
            obs_and_predict_single =\
                self.predict_positions(mcs_data_all, description)
            obs_and_predict_multi.append(obs_and_predict_single)
        self.obs_and_predict_multi = obs_and_predict_multi

    def predict_positions(self, mcs_data_all, type_of_run='theta',
                          outlier_distance=2, max_break_distance=1.7):
        """Predict positions of spots which were not observed

        Parameters
        ----------
        mcs_data_all: `np.array`, (2, N_fiber, N_obs)
           Positions of observed spots
        type_of_run: `str`
            Describing which kind of movement is being done
            {'theta', 'phi'}
        outlier_distance: `float`, optional
            Points more than outlier_distance from
            the median of all points will be replaced by nan [mm]
        max_break_distance: `float`, optional
            Maximal distance between points on the
            either side of the black dot [mm]

        Returns
        ----------
        prepared_observations_and_predictions `list`
            lists contaning 4 arrays, which contains
            observed positions in x and y,
            and predicted position in x and y. Each array has a
            length equal to the number of observations.

        Notes
        ----------
        Creates predictions for where the points should be,
        when you do not see points.

        The data is first cleaned. Points more than outlier_distance
        are removed as they have been atributed to wrong cobras.

        If the predicted movement is too unreasonable, we disregard
        that prediction. The diameter of a dot is 1.5 mm,
        so the distance can not be larger than that,
        or we would see the data on the other side of the dot.
        I have placed 1.7 mm as a limit
        to account for possible variations of size and measurment errors.

        After cleaning of the data, the algoritm searches to see
        if there are breaks in the observed data.

        Depending on if you see points from both sides of the black dots,
        and depending on the tpye of movement, it changes the
        order of complexity for the extrapolation/interpolation
        """
        valid_run = {'theta', 'phi'}
        if type_of_run not in valid_run:
            raise ValueError("results: status must be one of %r." % valid_run)

        # lists which will contain the positions which have been observed
        # 2 runs and 2 coordinates = 4 lists
        list_of_actual_position_x_observed = []
        list_of_actual_position_y_observed = []

        # lists which will contain the position which were not observed
        # and we predicted with this algorithm
        list_of_predicted_position_x_not_observed = []
        list_of_predicted_position_y_not_observed = []

        for i in range(0, self.number_of_fibers):

            try:
                x_measurments = mcs_data_all[0, i]
                y_measurments = mcs_data_all[1, i]
                n_obs = len(y_measurments)

                # index of the points which are observed
                isGood_init = np.isfinite(x_measurments) & np.isfinite(y_measurments)

                # Select the point that are more than outlier_distance (2 mm) away from median
                # of all of the points and replace them with np.nan
                # We assume these points have been attributed to wrong cobras
                xMedian = np.median(x_measurments[isGood_init])
                yMedian = np.median(y_measurments[isGood_init])
                large_outliers_selection =\
                    (np.abs(y_measurments[isGood_init] - yMedian) > outlier_distance) |\
                    (np.abs(x_measurments[isGood_init] - xMedian) > outlier_distance)

                # remove the large outliers from the selection of good points
                isGood = np.copy(isGood_init)
                isGood[isGood_init == 1] = isGood_init[isGood_init == 1]*~large_outliers_selection

                # replace all unsuccessful measurments wih nan
                x_measurments[~isGood] = np.nan
                y_measurments[~isGood] = np.nan

                if np.sum(np.diff(isGood)) == 1:
                    number_of_breaks = 1
                else:
                    number_of_breaks = 2

                # if np.sum(np.diff(idx)) == 1 we see points on only one side of the black dot
                if type_of_run == 'theta':
                    poly_order = 3
                if type_of_run == 'phi':
                    poly_order_interpolation = 1
                # create prediction for where the points should be, when you do not see points
                # if you do not see points from both sides of the black dots, do simpler extrapolation
                # if you see points from both sides of the black dots, do more complex interpolation
                if number_of_breaks == 1:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        deg2_fit_to_single_point_y, residuals_y =\
                            np.polyfit(np.arange(0, n_obs)[isGood],
                                       y_measurments[isGood], poly_order, full=True)[0:2]
                        deg2_fit_to_single_point_x, residuals_x =\
                            np.polyfit(np.arange(0, n_obs)[isGood],
                                       x_measurments[isGood], poly_order, full=True)[0:2]
                else:
                    deg2_fit_to_single_point_y, residuals_y =\
                        np.polyfit(np.arange(0, n_obs)[isGood],
                                   y_measurments[isGood], poly_order + 2, full=True)[0:2]
                    deg2_fit_to_single_point_x, residuals_x =\
                        np.polyfit(np.arange(0, n_obs)[isGood],
                                   x_measurments[isGood], poly_order + 2, full=True)[0:2]

                if len(residuals_y) == 0:
                    residuals_y = np.array([99])
                if len(residuals_x) == 0:
                    residuals_x = np.array([99])

                poly_deg_x = np.poly1d(deg2_fit_to_single_point_x)
                poly_deg_y = np.poly1d(deg2_fit_to_single_point_y)

                # some cleaning, to exclude the results where the interpolation results make no sense

                # if cobra has more than 5 observations hidden, calculate how much distance
                # we predicted that the cobra has moved during the `predicted` movement,
                # i.e., while it was not observed. This will be used
                # below to remove prediction that make no sense
                x_predicted = poly_deg_x(np.arange(0, n_obs))[~isGood]
                y_predicted = poly_deg_y(np.arange(0, n_obs))[~isGood]
                if np.sum(~isGood) > 5:
                    r_distance_predicted = np.hypot(x_predicted[-1] - x_predicted[0],
                                                    y_predicted[-1] - y_predicted[0])
                else:
                    r_distance_predicted = 0

                # Residuals from the polyfit are larger than 0.5 mm only in cases of
                # catastrophic failure
                if residuals_y[0] < 0.5 and residuals_x[0] < 0.5\
                        and ~((number_of_breaks == 1) and (r_distance_predicted > max_break_distance)):
                    predicted_position_x = poly_deg_x(np.arange(0, n_obs))
                    predicted_position_y = poly_deg_y(np.arange(0, n_obs))
                else:
                    predicted_position_x = np.full(n_obs, np.nan)
                    predicted_position_y = np.full(n_obs, np.nan)

                # create lists which contain observed and predicted data
                actual_position_x_observed = mcs_data_all[0, i][isGood]
                actual_position_y_observed = mcs_data_all[1, i][isGood]

                predicted_position_x_not_observed = predicted_position_x[~isGood]
                predicted_position_y_not_observed = predicted_position_y[~isGood]

                list_of_actual_position_x_observed.append(actual_position_x_observed)
                list_of_actual_position_y_observed.append(actual_position_y_observed)

                list_of_predicted_position_x_not_observed.append(predicted_position_x_not_observed)
                list_of_predicted_position_y_not_observed.append(predicted_position_y_not_observed)
            except (TypeError, ValueError):
                # if the process failed at some points, do not consider this fiber in the later stages
                list_of_actual_position_x_observed.append([])
                list_of_actual_position_y_observed.append([])

                list_of_predicted_position_x_not_observed.append(np.full(n_obs, np.nan))
                list_of_predicted_position_y_not_observed.append(np.full(n_obs, np.nan))

        prepared_observations_and_predictions = [list_of_actual_position_x_observed,
                                                 list_of_actual_position_y_observed,
                                                 list_of_predicted_position_x_not_observed,
                                                 list_of_predicted_position_y_not_observed]

        return prepared_observations_and_predictions

    @staticmethod
    def check(x, y, xc, yc, r):
        """Check if the cobra is covered by the black dot

        Parameters
        ----------
        x: `float`
            x coordinates of the dot [mm]
        y: `float`
            y coordinate of the dot [mm]
        xc: `float`
            x coordinate of the cobra [mm]
        yc: `float`
            y coordinate of the cobra [mm]
        r: `float`
            radius of dot [mm]

        Returns
        ----------
        cover value: `bool`
            returns True if the cobra is covered

        Notes
        ----------
        Gets called by `quality_measure_single_spot`
        """

        return ((x - xc)**2 + (y - yc)**2) < r**2

    def new_position_of_dot(self, i, xd_mod, yd_mod, scale, rot, x_scale_rot, y_scale_rot):
        """Move the dot to the new position

        Parameters
        ----------
        i: `integer`
            index of the cobra, starting from 1
        xd_mod: `float`
            x offset of the new position [mm]
        yd_mod: `float`
            y offset of the new position [mm]
        scale: `float`
            overall scaling
        x_scale_rot: `float`
            x coordinate of the point
            around which to rotate and scale [mm]
        x_scale_rot: `float`
            y coordinate of the point
            around which to rotate and scale[mm]

        Returns
        ----------
        xd_new, yd_new: `float`, `float`
            x and y coordinate of the new position [mm]

        Notes
        ---------
        This function allows the user to move the dots in x and y direction,
        and rotate/scale around custom origin

        Calls `rotate`
        """
        dots = self.dots

        xd_original = dots.loc[i-1]['x']
        yd_original = dots.loc[i-1]['y']

        xd_offset = xd_original - x_scale_rot
        yd_offset = yd_original - y_scale_rot

        r_distance_from_center = np.hypot(xd_offset, yd_offset)

        xc_rotated, yc_rotated = rotate((xd_offset*(1+scale*r_distance_from_center/100),
                                         yd_offset*(1+scale*r_distance_from_center/100)), (0, 0), rot)

        xc_rotated = xc_rotated+x_scale_rot
        yc_rotated = yc_rotated+y_scale_rot

        xd_new = xc_rotated + xd_mod
        yd_new = yc_rotated + yd_mod
        return xd_new, yd_new

    def quality_measure_single_spot(self, obs_and_predict_multi_single_fiber,
                                    xd, yd):
        """Calculate a penalty given x and y dot positions

        Parameters
        ----------
        obs_and_predict_multi: `list`
            list contaning observations for only one fiber
        xd: `float`
            x position of the dot [mm]
        yd: `float`
            y position of the dot [mm]

        Returns
        ----------
        penalty_measure: `float`
            total penalty for this position of the dot

        Notes
        ---------
        Calculates `penalty`. Penalty consist of 2 factors:
            1. Points should have been seen given the dot position,
            but they are not seen
            2. Point which are seen, but they should not have been
            seen, given the dot position

        Calls `check`
        Gets called by `total_penalty_for_single_dot`
        """
        list_of_penalty_measure_single_run = []
        for i in range(len(obs_and_predict_multi_single_fiber)):
            prepared_observations_and_predictions =\
                obs_and_predict_multi_single_fiber[i]

            actual_position_x_observed,\
                actual_position_y_observed,\
                predicted_position_x_not_observed,\
                predicted_position_y_not_observed = prepared_observations_and_predictions

            # the ones which were seen - add penalty if you should not have seen them
            penalty_measure_observed = []
            for j in range(len(actual_position_x_observed)):
                penalty_measure_observed.append(self.check(actual_position_x_observed[j],
                                                           actual_position_y_observed[j],
                                                           xd, yd, self.radius_of_black_dots))

            # the ones which should not be seen - add penalty if you saw them
            penalty_measure_predicted = []
            for j in range(len(predicted_position_x_not_observed)):
                # if you were not able to predict position, no penalty
                if np.isnan(predicted_position_x_not_observed[j]):
                    val = 0
                else:
                    val = self.check(predicted_position_x_not_observed[j],
                                     predicted_position_y_not_observed[j],
                                     xd, yd, self.radius_of_black_dots)
                    val = 1 - val
                penalty_measure_predicted.append(val)

            penalty_measure_single_run = np.sum(penalty_measure_observed) + np.sum(penalty_measure_predicted)
            list_of_penalty_measure_single_run.append(penalty_measure_single_run)

        sum_penalty_measure_single_run = np.sum(list_of_penalty_measure_single_run)

        return sum_penalty_measure_single_run

    def total_penalty_for_single_dot(self, i, xd, yd):
        """Calculate a penalty for a single dot

        Wrapper function for `quality_measure_single_spot`, that
        calculate penatly given index of the spot and the dot position

        Parameters
        ----------
        i: `integer`
            index of the cobra
        xd: `float`
            x position of the dot [mm]
        yd: `float`
            y position of the dot [mm]

        Returns
        ----------
        total_penalty_for_single_modification: `float`
            total penalty for this position of the dot
        """
        obs_and_predict_multi_single_fiber = []
        for obs in range(self.n_of_obs):
            obs_and_predict_multi_single_fiber_single_run = []
            for j in range(4):
                obs_and_predict_multi_single_fiber_single_run.append(
                    self.obs_and_predict_multi[obs][j][i])

            obs_and_predict_multi_single_fiber.append(
                obs_and_predict_multi_single_fiber_single_run)

        single_run_result_for_modification = []
        try:
            single_run_result_for_modification.\
                append([i, self.quality_measure_single_spot(
                    obs_and_predict_multi_single_fiber, xd, yd)])
        except (TypeError, ValueError):
            pass

        single_run_result_for_modification = np.array(single_run_result_for_modification)

        total_penalty_for_single_modification = np.sum(single_run_result_for_modification[:, 1])
        return total_penalty_for_single_modification

    def optimize_function(self, design_variables, return_full_result=False):
        """Find penalty for all of fibers given the suggested moves.

        This function output needs to be optimized in order to
        get best fit ot the data.

        Parameters
        ----------
        design_variables: `np.array` of `floats`
            variables that describe the transformation
        return_full_result: `bool`
            change the amount of output

        Returns
        ----------
        if return_full_result == False:
           single float with total penalty
        if return_full_result == True:
           array with all dots, where each line contains [index,
           penalty before optimization, penalty after optimization]

        Notes
        ----------
        Takes the default position of black dots and
        applies simple tranformations to these positions.
        Calculates the `penalty', i.e., how well do
        transformed black dots describe the reality.

        Calls `new_position_of_dot`
        Calls `total_penalty_for_single_dot`

        Examples:
        ----------
        # Pass this function to minimization routine.
        # For November 2021 run I used:
        >>>> bounds = np.array([(-0.5,0.5), (-0.5,0.5), (-0.005,0.005),
                           (-0.2,0.2), (-200,200), (-200,200)])
        # spawn the initial simples to search for the solutions
        >>>> init_simplex = np.zeros((7, 6))
        >>>> for l in range(1,7):
        >>>>     init_simplex[l] = bounds[:, 0]+(bounds[:, 1]-bounds[:, 0])*np.random.random_sample(size=6)
        >>>> init_simplex[0]=[  0.23196286,  -0.11102326,  -0.00062335,
                         -0.00675098, -78.35643258,  69.39903929]

        # simple Nelder-Mead will do sufficently good job
        >>>> res = scipy.optimize.minimize(optimize_black_dots_instance.optimize_function,
                                      x0=init_simplex[0], method='Nelder-Mead',
                                      options={'maxiter':1000, 'initial_simplex':init_simplex})
        """
        xd_mod = design_variables[0]
        yd_mod = design_variables[1]
        scale = design_variables[2]
        rot = design_variables[3]
        x_scale_rot = design_variables[4]
        y_scale_rot = design_variables[5]

        list_of_test_dots = []
        list_of_total_penalty_for_single_dot = []
        for i in np.arange(0+1, self.number_of_fibers+1):
            try:
                new_position_for_single_dot = self.new_position_of_dot(i, xd_mod, yd_mod,
                                                                       scale, rot, x_scale_rot, y_scale_rot)
                list_of_total_penalty_for_single_dot.append(
                    self.total_penalty_for_single_dot(i, new_position_for_single_dot[0],
                                                      new_position_for_single_dot[1]))
                list_of_test_dots.append(i)
            except IndexError:
                pass
        if return_full_result is False:
            return np.sum(list_of_total_penalty_for_single_dot)
        else:
            return np.array(list_of_test_dots), list_of_total_penalty_for_single_dot


def rotate(point, origin, angle):
    """Rotate the point around the origin

    Parameters
    ----------
    point: `tuple`
        x, y coordinate of the point to rotate
    origin: `tuple`
        x, y coordinate of the point around which to rotate
    angle: `float`
        rotation angle [degrees]

    Returns
    ----------
    x, y: `float`, `float`
        x, y coordinate of the rotated point
    """
    radians = np.deg2rad(angle)
    x, y = point
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return qx, qy
