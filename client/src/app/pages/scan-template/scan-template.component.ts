import { Component } from '@angular/core';
import { FileUploaderComponent } from "../../components/file-uploader/file-uploader.component";
import { ActivatedRoute } from '@angular/router';
import { ScanTemplateService } from './scan-template.service';

@Component({
  selector: 'app-scan-template',
  standalone: true,
  imports: [FileUploaderComponent],
  templateUrl: './scan-template.component.html',
})
export class ScanTemplateComponent {
  constructor(private route: ActivatedRoute, private scanTemplateService: ScanTemplateService) {}
  public pageId: string | null = '';
  public images: any[] = [];

  ngOnInit(): void {
    this.route.paramMap.subscribe(params => {
      this.pageId = params.get('id');
      if (this.pageId) {
        this.loadImages(this.pageId);
      }
    });
  }

  private loadImages(pageId: string) {
//to do
  }
}
