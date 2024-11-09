import { Component } from '@angular/core';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [],
  templateUrl: './header.component.html',
  styles: ``
})
export class HeaderComponent {
  public scanConfig: { url: string, title: string} [] = [
    {url: 'home', title: 'Dashboard'}, 
    {url: 'home', title: 'About us'}, 
    {url: 'home', title: 'Contact'} ];
}
