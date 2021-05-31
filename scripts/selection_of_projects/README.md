# Selection of Projects

The notebooks on this folder show:

- the cleaning and pre-processing of the project's data collected from the GitHub API v4.
- descriptive statistics and attributes of the selected projects and the dataset

We collected from GitHub API data about the 2731 projects of the dataset of Ghiotto 2018 [paper](https://ieeexplore.ieee.org/abstract/document/8468085).
The API informed that 134 projects were not found (probably their owners removed it from GitHub) and 243 projects had their *owner/name* changed.
Therefore, for the projects with new *owner/name*, we had to map the old *owner/name* stored at the original 2018 dataset with the new *owner/name* we found on 2021. This was necessary to join the dataset with the new data collected from the GitHub API.

We selected projects with at least 1000 chunks, resulting in 29 projects.
A chunk is a region marked in a file, indicating there is a merge conflict to be solved.

## Notebooks summary

<table>
  <thead>
    <tr>
      <th>Notebook</th>
      <th>Input</th>
      <th>Output</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>github_api_data_preprocess.ipynb</td>
      <td> 
        <ul>
          <li>../data/number_conflicting_chunks.csv</li>
          <li>../data/number_chunks__updated_repos.csv</li>
        </ul>
      </td>
      <td>../data/api_data.csv</td>
      <td>Pre-processing of GitHub API data and joining it with with #chunks data from database</td>
    </tr>
    <tr>
      <td>selected_projects.ipynb</td>
      <td>../data/api_data.csv</td>
      <td> 
        <ul>
          <li>Descriptive statistics of the selected projects and the dataset</li>
          <li>Histograms of the selected projects and the dataset</li>
          <li>Scatterplot matrix of the selected projects and the dataset</li>
        </ul>
      </td>
      <td>Descriptive statistics, histograms and scatterplots of both the selected projects and the dataset</td>
    </tr>
  </tbody>
</table>
