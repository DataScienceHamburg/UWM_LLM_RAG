from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, WebsiteSearchTool

# Uncomment the following line to use an example of a custom tool
# from news_analysis.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

#%% Tools
search_tool = SerperDevTool()
website_search_tool = WebsiteSearchTool()

@CrewBase
class NewsAnalysis():
	"""NewsAnalysis crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			tools=[search_tool, website_search_tool], # Example of custom tool, loaded on the beginning of file
			verbose=True
		)
	
	@agent
	def analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['analyst'],
			# tools=[MyCustomTool()], # Example of custom tool, loaded on the beginning of file
			verbose=True
		)

	@agent
	def writer(self) -> Agent:
		return Agent(
			config=self.agents_config['writer'],
			verbose=True
		)

	@task
	def information_gathering_task(self) -> Task:
		return Task(
			config=self.tasks_config['information_gathering_task'],
		)

	@task
	def fact_checking_task(self) -> Task:
		return Task(
			config=self.tasks_config['fact_checking_task'],
			output_file='report.md'
		)
  
	@task
	def context_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['context_analysis_task'],
		)
  
	@task
	def report_assembly_task(self) -> Task:
		return Task(
			config=self.tasks_config['report_assembly_task'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the NewsAnalysis crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
