import { Component } from '@angular/core';
import { RouterLink, RouterModule, RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-header',
  standalone: true,
  imports:[RouterModule],
  templateUrl: './header.component.html',
  styles: ``
})
export class HeaderComponent {
  public scanConfig: { url: string, title: string} [] = [
    {url: 'dashboard', title: 'Dashboard'}, 
    {url: 'home', title: 'About us'}, 
    {url: 'home', title: 'Contact'} ];
}
