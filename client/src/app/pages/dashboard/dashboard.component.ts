import { Component } from '@angular/core';
import { RouterModule } from '@angular/router';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [RouterModule],
  templateUrl: './dashboard.component.html',
  styles: ``
})
export class DashboardComponent {
  public modelPages: {name: string, description: string, path: string}[] = [
      {
        name: 'Retinal imaging',
        description: 'Explore advanced retinal imaging techniques for early detection and monitoring of eye diseases, providing insights into ocular and systemic health.',
        path: 'retinal-imaging'
      },    
      {
        name: 'Kidney cancer',
        description: 'Detailed analysis of kidney cancer models, including diagnostics, staging, and treatment options to support informed clinical decision-making.',
        path: 'kidney-cancer'
      },    
      {
        name: 'Alzheimer',
        description: 'Comprehensive models for Alzheimer’s disease, focusing on progression, risk factors, and early intervention strategies for cognitive health.',
        path: 'alzheimer'
      },    
      {
        name: 'Lung Cancer',
        description: 'Insights into lung cancer diagnosis and prognosis, featuring models that assist in evaluating stages, treatment responses, and survival outcomes.',
        path: 'lung-cancer'
      },    
      {
        name: 'Covid',
        description: 'Covid-19 models addressing diagnostics, infection trends, and impact assessments to support effective public health responses.',
        path: 'covid'
      },
      {
        name: 'Malaria',
        description: 'Malaria is a serious infectious disease caused by parasites transmitted to humans through the bites of infected Anopheles mosquitoes. ',
        path: 'malaria'
      },
      {
        name: 'Segmentation',
        description: 'Segmentation is the process of partitioning a digital image into multiple segments to simplify the representation of an image.',
        path: 'segmentation'
      }
    ]    
}
