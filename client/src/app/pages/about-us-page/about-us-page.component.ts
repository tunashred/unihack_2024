import { Component } from '@angular/core';

@Component({
  selector: 'app-about-us-page',
  standalone: true,
  imports: [],
  templateUrl: './about-us-page.component.html',
  styles: ``
})
export class AboutUsPageComponent {
  public members: {name: string, github: string, link: string, image: string, pronouns: string, role: string, motto: string}[] = [
    {
      name: 'Șoitu Viorel', github: 'https://github.com/1viorel', 
      link: 'https://1viorel.tech',
      image: 'https://avatars.githubusercontent.com/u/32220246?v=4',
      pronouns: 'drop/table',
      role: 'Web developer',
      motto: 'Can center a div, did the frontend and some of the backend.'
    },
    {
      name: 'Adrian Badea', github: '', 
      link: 'https://cdn.discordapp.com/attachments/1304489004636573700/1304748692946817095/terrorist-face-but-made-in-source-2-v0-cuvb9467grrb1.png?ex=67308569&is=672f33e9&hm=959f6a084ba60d08d9e8c13ec783f37a0d8ec826e2f3c6215f8f1b022c5fb9a8&',
      image: '',
      pronouns: 'BET/MAN',
      role: '',
      motto: ''
    },
    {
      name: 'Alex Enache', github: '', 
      link: '',
      image: '',
      pronouns: '',
      role: '',
      motto: ''
    },
    {
      name: 'Enache', github: '', 
      link: '',
      image: '',
      pronouns: '',
      role: '',
      motto: ''
    },
    ]
}
